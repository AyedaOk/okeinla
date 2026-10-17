import unittest
import sys
import random
import json
import tempfile
import zipfile
from pathlib import Path
import numpy as np
import torch

sys.path.append("..")

from rawnind.libs import raw
from rawnind.libs import rawproc
from rawnind.libs.abstract_trainer import (
    BayerImageToImageNN,
    BayerImageToImageNNTraining,
    ImageToImageNN,
)
from rawnind.models import nafnet, raw_denoiser
from rawnind.tools import export_darktable_model
from common.libs import pt_losses


class TestRawproc(unittest.TestCase):
    def test_bayer_training_step_updates_parameters(self):
        trainer = object.__new__(BayerImageToImageNNTraining)
        trainer.device = torch.device("cpu")
        trainer.debug_options = []
        trainer.match_gain = "output"
        trainer.transfer = torch.nn.Identity()
        trainer.lossf = torch.nn.MSELoss()
        trainer.model = nafnet.NAFNet(
            in_channels=4, funit=4, enc_blk_nums=(1,), middle_blk_num=1, dec_blk_nums=(1,)
        )
        optimizer = torch.optim.Adam(trainer.model.parameters(), lr=1e-4)
        batch = {
            "x_crops": torch.rand(1, 1, 3, 32, 32),
            "y_crops": torch.rand(1, 1, 4, 16, 16),
            "mask_crops": torch.ones(1, 1, 3, 32, 32),
            "rgb_xyz_matrix": torch.cat((torch.eye(3), torch.zeros(1, 3))).unsqueeze(0),
        }
        before = next(trainer.model.parameters()).detach().clone()
        loss = trainer.step((batch, batch), optimizer)
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(torch.equal(before, next(trainer.model.parameters())))

    def test_demosaic_preserves_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image = torch.rand(1, 4, 16, 16, device=device)
        output = rawproc.demosaic(image)
        self.assertEqual(output.device.type, device.type)
        self.assertEqual(output.shape, (1, 3, 32, 32))

    def test_reproducibility_seed(self):
        ImageToImageNN.configure_reproducibility(seed=2026, deterministic=False)
        first = (random.random(), np.random.random(), torch.rand(1).item())
        ImageToImageNN.configure_reproducibility(seed=2026, deterministic=False)
        second = (random.random(), np.random.random(), torch.rand(1).item())
        self.assertEqual(first, second)

    def test_learning_rate_decay_applied_once(self):
        trainer = object.__new__(BayerImageToImageNNTraining)
        parameter = torch.nn.Parameter(torch.ones(()))
        trainer.optimizer = torch.optim.Adam([parameter], lr=0.0003)
        trainer.best_validation_losses = {"rawforge_loss": 1.0}
        trainer.lr_adjustment_allowed_step = 0
        trainer.lr_multiplier = 0.85
        trainer.patience = 100

        class ResultSaver:
            def add_res(self, *_args, **_kwargs):
                pass

        trainer.json_saver = ResultSaver()
        trainer.adjust_lr({"rawforge_loss": 2.0}, step=1)
        self.assertAlmostEqual(trainer.optimizer.param_groups[0]["lr"], 0.000255)

    def test_stop_request_before_training_step(self):
        trainer = object.__new__(BayerImageToImageNNTraining)
        trainer.stop_requested = True
        trainer.debug_options = []
        loss = trainer.train(None, 1, [None], [None])
        self.assertTrue(np.isnan(loss))
        self.assertEqual(trainer.last_num_training_steps, 0)

    def test_full_training_pass_uses_legacy_nominal_steps(self):
        nominal_steps = BayerImageToImageNNTraining.get_nominal_training_steps(
            requested_steps=5000,
            completed_updates=1046,
            pass_capacity=1046,
            interrupted=False,
        )
        self.assertEqual(nominal_steps, 5000)

    def test_interrupted_training_pass_scales_nominal_steps(self):
        nominal_steps = BayerImageToImageNNTraining.get_nominal_training_steps(
            requested_steps=5000,
            completed_updates=523,
            pass_capacity=1046,
            interrupted=True,
        )
        self.assertEqual(nominal_steps, 2500)

    def test_l1_msssim_loss(self):
        torch.manual_seed(2026)
        prediction = torch.rand(1, 3, 256, 256, requires_grad=True)
        target = torch.rand_like(prediction)
        loss_function = pt_losses.L1MSSSIMLoss()

        loss = loss_function(prediction, target)
        expected = 0.16 * torch.nn.functional.l1_loss(
            prediction, target
        ) + 0.84 * pt_losses.MS_SSIM_loss()(prediction, target)
        self.assertTrue(torch.allclose(loss, expected))

        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertAlmostEqual(loss_function(target, target).item(), 0.0, places=6)

    def test_shadow_l1_msssim_loss(self):
        loss_function = pt_losses.ShadowL1MSSSIMLoss()
        dark_target = torch.zeros(1, 3, 16, 16)
        bright_target = torch.ones_like(dark_target)
        dark_prediction = dark_target + 0.1
        bright_prediction = bright_target - 0.1
        self.assertGreater(
            loss_function.shadow_weighted_l1(dark_prediction, dark_target),
            loss_function.shadow_weighted_l1(bright_prediction, bright_target),
        )

        torch.manual_seed(2026)
        prediction = torch.rand(1, 3, 256, 256, requires_grad=True)
        target = torch.rand_like(prediction)
        loss = loss_function(prediction, target)
        expected = 0.16 * loss_function.shadow_weighted_l1(
            prediction, target
        ) + 0.84 * pt_losses.MS_SSIM_loss()(prediction, target)
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_rawforge_loss(self):
        torch.manual_seed(2026)
        prediction = torch.rand(2, 3, 256, 256, requires_grad=True)
        target = torch.rand_like(prediction)
        perceptual_loss = torch.nn.L1Loss()
        loss_function = pt_losses.RawForgeLoss(
            perceptual_loss=perceptual_loss, perceptual_chunk_size=1
        )

        prediction_gamma = loss_function.apply_gamma(prediction)
        target_gamma = loss_function.apply_gamma(target)
        expected = (
            0.16
            * loss_function.shadow_weighted_l1(prediction_gamma, target_gamma)
            + 0.84
            * pt_losses.MS_SSIM_loss()(prediction_gamma, target_gamma)
            + 0.025 * perceptual_loss(prediction_gamma, target_gamma)
        )
        loss = loss_function(prediction, target)
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertAlmostEqual(loss_function(target, target).item(), 0.0, places=6)

    def test_rawforge_detail_loss_changes_only_perceptual_weight(self):
        perceptual = torch.nn.L1Loss()
        baseline = pt_losses.RawForgeLoss(perceptual_loss=perceptual)
        detail = pt_losses.losses["rawforge_detail_loss"](perceptual_loss=perceptual)
        torch.manual_seed(2026)
        prediction = torch.rand(1, 3, 256, 256)
        target = torch.rand_like(prediction)
        extra = 0.01 * perceptual(baseline.apply_gamma(prediction), baseline.apply_gamma(target))
        self.assertEqual(detail.perceptual_weight, 0.035)
        self.assertTrue(torch.allclose(detail(prediction, target) - baseline(prediction, target), extra))

    def test_iso_normalization(self):
        iso = torch.tensor([100.0, 800.0, 51200.0])
        expected = torch.tensor([0.0, 1.0 / 3.0, 1.0])
        self.assertTrue(torch.allclose(BayerImageToImageNN.normalize_iso(iso), expected))
        with self.assertRaises(ValueError):
            BayerImageToImageNN.normalize_iso(torch.tensor([0.0]))

    def test_iso_conditioned_unet(self):
        model = raw_denoiser.UtNet2(
            in_channels=4, funit=4, conditioning_channels=1
        )
        image = torch.rand(2, 4, 32, 32)
        output = model(image, conditioning=torch.tensor([0.0, 1.0]))
        self.assertEqual(output.shape, (2, 3, 64, 64))
        with self.assertRaises(ValueError):
            model(image)

    def test_iso_repack_batch(self):
        def make_batch(batch_size, iso):
            return {
                "x_crops": torch.rand(batch_size, 2, 3, 16, 16),
                "y_crops": torch.rand(batch_size, 2, 4, 8, 8),
                "mask_crops": torch.ones(batch_size, 2, 3, 16, 16),
                "rgb_xyz_matrix": torch.rand(batch_size, 4, 3),
                "iso": torch.tensor(iso),
            }

        repacked = BayerImageToImageNNTraining.repack_batch(
            (make_batch(1, [100.0]), make_batch(2, [200.0, 400.0])),
            torch.device("cpu"),
        )
        self.assertEqual(repacked["y_crops"].shape, (6, 4, 8, 8))
        self.assertTrue(
            torch.equal(
                repacked["iso"],
                torch.tensor([100.0, 100.0, 200.0, 200.0, 400.0, 400.0]),
            )
        )

    def test_bayer_nafnet_geometry_and_gradients(self):
        model = nafnet.NAFNet(
            in_channels=4,
            funit=4,
            enc_blk_nums=(1, 1),
            middle_blk_num=1,
            dec_blk_nums=(1, 1),
        )
        image = torch.rand(1, 4, 31, 37, requires_grad=True)
        output = model(image)
        self.assertEqual(output.shape, (1, 3, 62, 74))
        output.mean().backward()
        self.assertTrue(torch.isfinite(image.grad).all())

    def test_bayer_nafnet_parameter_count(self):
        model = nafnet.NAFNet(in_channels=4, funit=16)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 7373372)
        wide_model = nafnet.NAFNet(in_channels=4, funit=32)
        self.assertEqual(
            sum(parameter.numel() for parameter in wide_model.parameters()), 29162604
        )

    def test_darktable_export_model_factory(self):
        args = {
            "arch": "unet",
            "funit": 4,
            "in_channels": 4,
            "preupsample": False,
            "iso_conditioning": False,
            "match_gain": "output",
            "transfer_function": "None",
        }
        unet = export_darktable_model.build_model(args)
        self.assertEqual(unet(torch.rand(1, 4, 16, 16)).shape, (1, 3, 32, 32))

        args["arch"] = "nafnet"
        naf_model = export_darktable_model.build_model(args)
        self.assertEqual(
            naf_model(torch.rand(1, 4, 16, 16)).shape, (1, 3, 32, 32)
        )

        args["arch"] = "unet"
        args["iso_conditioning"] = True
        with self.assertRaisesRegex(ValueError, "does not provide the ISO input"):
            export_darktable_model.build_model(args)

    def test_darktable_checkpoint_selection_and_package(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model_dir = root / "experiment"
            checkpoint_dir = model_dir / "saved_models"
            checkpoint_dir.mkdir(parents=True)
            checkpoint = checkpoint_dir / "iter_42.pt"
            checkpoint.write_bytes(b"checkpoint")
            (model_dir / "trainres.yaml").write_text(
                "best_step:\n  val_rawforge_loss.None: 42\n"
            )
            selected = export_darktable_model.resolve_checkpoint(
                model_dir, None, export_darktable_model.DEFAULT_METRIC
            )
            self.assertEqual(selected, checkpoint)

            args = {"arch": "nafnet", "expname": "M4A Test"}
            config = export_darktable_model.make_config(
                args,
                "m4a-test",
                "M4A Test",
                "Test model",
                "1.0",
                512,
                checkpoint,
            )
            self.assertEqual(config["arch"], "nafnet")
            args["arch"] = "unet"
            self.assertEqual(
                export_darktable_model.make_config(
                    args,
                    "m2c-test",
                    "M2C Test",
                    "Test model",
                    "1.0",
                    512,
                    checkpoint,
                )["arch"],
                "utnet2",
            )
            package_dir = root / "m4a-test"
            package_dir.mkdir()
            (package_dir / "config.json").write_text(json.dumps(config))
            (package_dir / "model_bayer.onnx").write_bytes(b"onnx")
            archive_path = root / "m4a-test.dtmodel"
            export_darktable_model.package_dtmodel(package_dir, archive_path)
            with zipfile.ZipFile(archive_path) as archive:
                self.assertEqual(
                    set(archive.namelist()),
                    {"m4a-test/config.json", "m4a-test/model_bayer.onnx"},
                )

    def test_scenelin_to_pq(self):
        """Test that scenelin_to_pq pytorch matches numpy colour_science library."""
        ptbatch = torch.rand(5, 3, 256, 256) * 1.1 - 0.1
        npbatch = ptbatch.numpy()
        nptransformed = rawproc.scenelin_to_pq(npbatch)
        pttransformed = rawproc.scenelin_to_pq(ptbatch)
        self.assertTrue(
            torch.allclose(
                pttransformed,
                torch.from_numpy(nptransformed).to(torch.float32),
                atol=1e-05,
            )
        )

    def test_match_gain(self):
        """Check that the gain is matched for 0 and 1 and that the dimensions remain the same for single and batched images."""
        anchor_single_0 = torch.zeros(3, 256, 256, dtype=torch.float32)
        anchor_single_1 = torch.ones_like(anchor_single_0)
        anchor_batch_0 = torch.zeros(5, 3, 256, 256, dtype=torch.float32)
        anchor_batch_1 = torch.ones_like(anchor_batch_0)
        rand_single = torch.rand_like(anchor_single_0)
        rand_batch = torch.rand_like(anchor_batch_0)
        matched_single_0 = rawproc.match_gain(
            anchor_img=anchor_single_0, other_img=rand_single
        )
        matched_single_1 = rawproc.match_gain(
            anchor_img=anchor_single_1, other_img=rand_single
        )
        matched_batch_0 = rawproc.match_gain(
            anchor_img=anchor_batch_0, other_img=rand_batch
        )
        matched_batch_1 = rawproc.match_gain(
            anchor_img=anchor_batch_1, other_img=rand_batch
        )
        self.assertAlmostEqual(matched_single_0.mean().item(), 0.0, places=5)
        self.assertAlmostEqual(matched_single_1.mean().item(), 1.0, places=5)
        self.assertAlmostEqual(matched_batch_0.mean().item(), 0.0, places=5)
        self.assertAlmostEqual(matched_batch_1.mean().item(), 1.0, places=5)
        self.assertAlmostEqual(matched_single_0.shape, (3, 256, 256))
        self.assertAlmostEqual(matched_batch_1.shape, (5, 3, 256, 256))

    def test_camRGB_to_rec2020_batch_conversion(self):
        """Ensure the output of rawproc.camRGB_to_lin_rec2020_images (pytorch batched) is the same as that of raw.camRGB_to_profiledRGB_img (numpy single)."""
        rgb_xyz_matrix_1 = torch.tensor(
            [
                [0.9943, -0.3269, -0.0839],
                [-0.5323, 1.3269, 0.2259],
                [-0.1198, 0.2083, 0.7557],
                [0.0000, 0.0000, 0.0000],
            ]
        )
        rgb_xyz_matrix_2 = torch.tensor(
            [
                [0.6988, -0.1384, -0.0714],
                [-0.5631, 1.3410, 0.2447],
                [-0.1485, 0.2204, 0.7318],
                [0.0000, 0.0000, 0.0000],
            ]
        )
        rgb_xyz_matrix_3 = torch.tensor(
            [
                [0.7888, -0.1902, -0.1011],
                [-0.8106, 1.6085, 0.2099],
                [-0.2353, 0.2866, 0.7330],
                [0.0000, 0.0000, 0.0000],
            ]
        )
        rgb_xyz_matrix_4 = rgb_xyz_matrix_2
        rgb_xyz_matrices = torch.stack(
            (rgb_xyz_matrix_1, rgb_xyz_matrix_2, rgb_xyz_matrix_3, rgb_xyz_matrix_4)
        )
        img1_camRGB = torch.rand(3, 256, 256)
        img2_camRGB = torch.rand(3, 256, 256)
        img3_camRGB = torch.rand(3, 256, 256)
        img4_camRGB = torch.rand(3, 256, 256)
        images_camRGB = torch.stack(
            (img1_camRGB, img2_camRGB, img3_camRGB, img4_camRGB)
        )
        img1_rec2020 = torch.from_numpy(
            raw.camRGB_to_profiledRGB_img(
                img1_camRGB.numpy(),
                {"rgb_xyz_matrix": rgb_xyz_matrix_1.numpy()},
                "lin_rec2020",
            )
        )
        img2_rec2020 = torch.from_numpy(
            raw.camRGB_to_profiledRGB_img(
                img2_camRGB.numpy(),
                {"rgb_xyz_matrix": rgb_xyz_matrix_2.numpy()},
                "lin_rec2020",
            )
        )
        img3_rec2020 = torch.from_numpy(
            raw.camRGB_to_profiledRGB_img(
                img3_camRGB.numpy(),
                {"rgb_xyz_matrix": rgb_xyz_matrix_3.numpy()},
                "lin_rec2020",
            )
        )
        img4_rec2020 = torch.from_numpy(
            raw.camRGB_to_profiledRGB_img(
                img4_camRGB.numpy(),
                {"rgb_xyz_matrix": rgb_xyz_matrix_4.numpy()},
                "lin_rec2020",
            )
        )
        images_rec2020 = rawproc.camRGB_to_lin_rec2020_images(
            images_camRGB, rgb_xyz_matrices
        )
        individual_outputs = torch.stack(
            (img1_rec2020, img2_rec2020, img3_rec2020, img4_rec2020)
        )

        self.assertTrue(torch.allclose(individual_outputs, images_rec2020, atol=1e-06))


if __name__ == "__main__":
    unittest.main()
