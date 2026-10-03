# Transformer calibration

Azure infrastructure for Qwen3-4B inference experiments. One Ubuntu Data Science Virtual Machine (DSVM) runs on **E8s_v5 for development** and resizes to **NC24ads_A100_v4 for experiments**, keeping its OS disk, Python environment, model cache, and results.

Defaults: South Central US, 256 GiB Standard SSD LRS, a static public IP, key-only SSH restricted to your address, and daily shutdown at 23:00 Denver time. The pinned image is `microsoft-dsvm:ubuntu-2204:2204-gen2:25.06.18`; Azure's South Central US catalog verified it as x64, Generation 2, with no Marketplace purchase plan on September 24, 2026. Actual CPU/GPU quota, allocation, and GPU runtime compatibility still need verification in your personal subscription.

The DSVM supplies the NVIDIA/CUDA software stack. Create your own project venv for PyTorch, Transformers, and experiment dependencies. There is no separate driver installer or Azure Machine Learning workspace in this configuration.

## Repository-local Azure configuration

Requires Azure CLI, Terraform >= 1.9, Python 3, and an SSH key. Run the following from the repository root:

```sh
cp -n VirtualMachine/terraform/personal.auto.tfvars.json.example VirtualMachine/terraform/personal.auto.tfvars.json
```

Edit `VirtualMachine/terraform/personal.auto.tfvars.json`:

- `subscription_id`: your new personal paid subscription ID.
- `allowed_ssh_source`: your public IPv4 address with `/32`.
- `ssh_public_key_path`: an existing public key path. To create a new key, run `ssh-keygen -t ed25519 -f ~/.ssh/transformer_calibration -C transformer-calibration` without overwriting an existing key.
- `vm_size`: keep `Standard_E8s_v5` initially.

For another trusted connection, add `additional_ssh_rules` to the same local JSON file. For example (replace the documentation IP with your hotspot's public IP):

```json
"additional_ssh_rules": {
  "MattHotSpot": { "source": "203.0.113.10/32", "priority": 1011 }
}
```

Keep a comma between JSON fields. Each additional rule permits TCP destination port 22 from that IP, using any source port. Use distinct priorities other than 1001, which belongs to the primary SSH rule. Update this configuration when your IP changes and regenerate the Terraform plan. Portal-only rules can be removed by a later Terraform apply because Terraform manages the security group's rules.

Terraform automatically reads this file. It is ignored by Git. Defaults live in `variables.tf`; add overrides such as `auto_shutdown_enabled` only when needed.

Use the repository's `cloud` helper for both tools:

```sh
./cloud az login
./cloud az account list --output table
./cloud az account show --query '{Name:name,SubscriptionId:id}' --output table
```

The helper sets `AZURE_CONFIG_DIR` to this repository's ignored `.azure/` directory. This holds a **separate login and selected subscription**, leaving the global Azure profile used by your other VM alone. Before ordinary Azure commands or authenticated Terraform commands, the helper selects the subscription from `personal.auto.tfvars.json`; if selection fails, it stops instead of using another subscription. Login and account listing work before that file is filled in, so you can discover your new subscription ID.

The helper also uses the local Azure profile for Terraform authentication and always runs Terraform in `VirtualMachine/terraform`. It disregards inherited `ARM_*` credentials so another project's service-principal settings cannot silently replace this CLI login. Explicit CLI flags or Terraform variable overrides remain your responsibility. Plain `az` commands outside the helper continue to use your normal global profile; Azure CLI does not automatically discover this repository's configuration.

Do not copy Terraform state from the other VM's repository. Keep this deployment's ignored state and plans in this checkout and back up the state securely.

## Check and deploy

Check the selected subscription and region before provisioning:

```sh
./cloud az vm image show --location southcentralus --urn microsoft-dsvm:ubuntu-2204:2204-gen2:25.06.18
./cloud az vm list-skus --location southcentralus --size Standard_E8s_v5 --all --output table
./cloud az vm list-skus --location southcentralus --size Standard_NC24ads_A100_v4 --all --output table
./cloud az vm list-usage --location southcentralus --output table
```

The A100 phase needs 24 available GPU-family vCPUs and 24 available total regional vCPUs. CPU development needs its own family quota. Quota is not a capacity reservation.

Provider auto-registration is disabled. Register the three services used by this configuration once, if not already registered:

```sh
./cloud az provider register --namespace Microsoft.Compute --wait
./cloud az provider register --namespace Microsoft.Network --wait
./cloud az provider register --namespace Microsoft.DevTestLab --wait
```

Then initialize, validate, and review the plan:

```sh
./cloud terraform init
./cloud terraform fmt -check
./cloud terraform validate
./cloud terraform plan -out=deployment.tfplan
```

The saved plan is written in `VirtualMachine/terraform`. After reviewing it, deploy and obtain the connection command:

```sh
./cloud terraform apply deployment.tfplan
./cloud terraform output -raw ssh_command
```

Keep notebooks behind an SSH tunnel. Use your own Jupyter process and token in the project venv rather than opening public notebook ports or relying on password-based DSVM JupyterHub login.

## Switch between CPU and GPU

Save all work before resizing. Change `vm_size` in `personal.auto.tfvars.json` to `Standard_NC24ads_A100_v4` for experiments or `Standard_E8s_v5` for development. Keep the image, region, and disk unchanged.

```sh
./cloud az vm deallocate --resource-group transformer-calibration-southcentralus-rg --name transformer-calibration-southcentralus-vm
./cloud terraform plan -out=resize.tfplan
./cloud terraform apply resize.tfplan
./cloud az vm start --resource-group transformer-calibration-southcentralus-rg --name transformer-calibration-southcentralus-vm
```

Review the resize plan before applying: the intended change is the VM size, not VM or disk replacement. If names were customized, use `terraform output resource_group` and `terraform output vm_name` through the helper to obtain them. Allocation can fail if the GPU is unavailable, even with approved quota.

At the first GPU boot, verify `nvidia-smi`, PyTorch CUDA access, and a 32K-context forward pass. Installed packages and cached downloads remain on the managed OS disk across resizes, but processes restart and the model must reload into memory. Put no authoritative files on temporary/local disks. Pin the working Python dependencies and keep thinking disabled.

Changing the image version or region can replace resources; an image change is not a routine resize. The configuration deliberately avoids `latest` so a later plan does not select a newer image during an unrelated size change.

## Shutdown and storage

Auto-shutdown deallocates the VM daily and does not restart it. Temporarily set `auto_shutdown_enabled` to `false` and apply before a long experiment, then restore it afterward. Save experiment progress frequently.

Deallocate when idle using the command above. Managed disks and the static public IP can continue billing while compute is deallocated. Destroying this deployment deletes the managed OS disk and its data; back up research files separately before any destruction.

See [RESOURCE_DEFINITION.md](docs/RESOURCE_DEFINITION.md) for the research configuration and [NEXT_STEPS.md](NEXT_STEPS.md) for the remaining setup and validation work.
