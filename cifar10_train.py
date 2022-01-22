# This is a Python script meant for training various CIFAR-10 models
# Unlike Fashion-MNIST, to achieve interesting results with a variety of models
# we require longer training times

# While there exist papers that improve on the runtimes, the best results we
# found so far is with long training times, hence this script

# PS: this is run as a SLURM job using GPUs provided by DoC

import attacks.pgd as pgd
import attacks.fgsm as fgsm
import argparse
import utils.dataloaders as dataloaders
import utils.clean_test as clean_test
import defenses.cifar10 as cifar10
import torch


# Parse the type of training
parser = argparse.ArgumentParser(
    description='Trains a DNN using the CIFAR-10 dataset.')
parser.add_argument('training_method', default="standard",
                    help='The training method for the model.')

args = parser.parse_args()
print("Using the following training method " + args.training_method)

# Actual implementation
DATA_ROOT = "./datasets/CIFAR10"

trainSetLoader, _, testSetLoader = dataloaders.get_CIFAR10_data_loaders(
    DATA_ROOT,
    trainSetSize=50000,
    validationSetSize=0,
    batchSize=128,
)

# In the case of this script, we want to simply save at root level (easier to grab)
SAVE_LOAD_ROOT = "."

# Possible attacks to use (well, the only useful ones so far)
attacks = {}

attacks["FGSM"] = fgsm.fgsm_attack
attacks["PGD"] = pgd.pgd_attack


# Depending on the training type, train and save a different model
if args.training_method == "standard":
    standard_model = cifar10.standard_training(
        trainSetLoader,
        load_if_available=True,
        load_path=SAVE_LOAD_ROOT + "/cifar10_standard",
    )

    # Test the model
    clean_test.test_trained_model(standard_model, testSetLoader)

    # Save the model
    torch.save(standard_model, SAVE_LOAD_ROOT + "/cifar10_standard")

if args.training_method == "fgsm":
    fgsm_model = cifar10.adversarial_training(
        trainSetLoader,
        "FGSM",
        attacks["FGSM"],
        load_if_available=True,
        load_path=SAVE_LOAD_ROOT + "/cifar10_fgsm",
        epsilon=0.05,
    )

    clean_test.test_trained_model(fgsm_model, testSetLoader)

    # Save the model
    torch.save(fgsm_model, SAVE_LOAD_ROOT + "/cifar10_fgsm")

if args.training_method == "pgd":
    pgd_model = cifar10.adversarial_training(
        trainSetLoader,
        "PGD",
        attacks["PGD"],
        load_if_available=True,
        load_path=SAVE_LOAD_ROOT + "/cifar10_pgd",
        epsilon=(8 / 255),
        alpha=(2 / 255),
        iterations=7,
    )

    clean_test.test_trained_model(pgd_model, testSetLoader)

    # Save the model
    torch.save(pgd_model, SAVE_LOAD_ROOT + "/cifar10_pgd")

if args.training_method == "interpolated_pgd":
    interpolated_pgd_model = cifar10.interpolated_adversarial_training(
        trainSetLoader,
        "PGD",
        attacks["PGD"],
        load_if_available=True,
        clip=True,
        load_path=SAVE_LOAD_ROOT + "/cifar10_interpolated_pgd",
        epsilon=(8 / 255),
        alpha=(2 / 255),
        iterations=7,
    )

    clean_test.test_trained_model(interpolated_pgd_model, testSetLoader)

    # Save the model
    torch.save(interpolated_pgd_model, SAVE_LOAD_ROOT +
               "/cifar10_interpolated_pgd")

if args.training_method == "dual":
    dual_model = cifar10.dual_adversarial_training(
        trainSetLoader,
        attacks["PGD"],
        attacks["FGSM"],
        load_if_available=True,
        load_path=SAVE_LOAD_ROOT + "/cifar10_dual",
        epsilon1=(8 / 255),
        epsilon2=0.1,
        alpha=(2 / 255),
        iterations=7,
    )

    clean_test.test_trained_model(dual_model, testSetLoader)

    # Save the model
    torch.save(dual_model, SAVE_LOAD_ROOT + "/cifar10_dual")

if args.training_method == "jacobian":
    jacobian_model = cifar10.jacobian_training(
        trainSetLoader,
        load_if_available=True,
        load_path=SAVE_LOAD_ROOT + "/cifar10_jacobian",
    )

    clean_test.test_trained_model(jacobian_model, testSetLoader)

    # Save the model
    torch.save(jacobian_model, SAVE_LOAD_ROOT + "/cifar10_jacobian")

if args.training_method == "alp":
    alp_model = cifar10.ALP_training(
        trainSetLoader,
        "PGD",
        attacks["PGD"],
        load_if_available=True,
        load_path=SAVE_LOAD_ROOT + "/cifar10_alp",
        epsilon=(8 / 255),
        alpha=(2 / 255),
        iterations=7,
    )

    clean_test.test_trained_model(alp_model, testSetLoader)

    # Save the model
    torch.save(alp_model, SAVE_LOAD_ROOT + "/cifar10_alp")

if args.training_method == "jacobian_alp":
    jacobian_alp_model = cifar10.jacobian_ALP_training(
        trainSetLoader,
        "PGD",
        attacks["PGD"],
        load_if_available=True,
        load_path=SAVE_LOAD_ROOT + "/cifar10_jacobian_alp",
        epsilon=(8 / 255),
        alpha=(2 / 255),
        iterations=7,
    )

    clean_test.test_trained_model(jacobian_alp_model, testSetLoader)

    # Save the model
    torch.save(jacobian_alp_model, SAVE_LOAD_ROOT + "/cifar10_jacobian_alp")