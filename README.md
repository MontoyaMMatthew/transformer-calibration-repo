# Transformer calibration

Personal Azure infrastructure for transformer calibration experiments. Adapted from the VM infrastructure in `RAI_704/my-rapid-ai-repo`, with a fresh Git history. Only Terraform configuration and its provider lock file were retained; class documentation, provisioning scripts, saved outputs, and Git metadata were excluded.

This provisions a clean Ubuntu 22.04 CPU VM, a virtual network/subnet, a static Standard public IP, an SSH-only security group, and daily auto-shutdown. No classroom software or experiment dependencies are installed. Install project dependencies explicitly when the experiment environment is defined. GPU drivers and CUDA are not configured.

## Configuration

Requires Terraform >= 1.9, Azure CLI, and an SSH key. The existing AzureRM 4.14 provider lock is retained for reproducibility.

```sh
cd VirtualMachine/terraform
cp personal.tfvars.json.example personal.tfvars.json
ssh-keygen -t ed25519 -f ~/.ssh/transformer_calibration -C transformer-calibration
az login
az account list --output table
```

Edit the ignored `personal.tfvars.json`:

- Replace `REDACTED` with the ID of your own Azure for Students or personal subscription.
- Set `allowed_ssh_source` to your public IPv4 address with `/32`. Update it when your network changes.
- Choose `location` and `vm_size` supported by that subscription. `eastus` and `Standard_D2s_v5` are starting values, not availability or free-tier guarantees.
- Adjust disk size and shutdown time as needed. Default shutdown is 23:00 Denver time (`Mountain Standard Time`, including daylight saving).

Resource names derive from the project and actual region, e.g. `transformer-calibration-eastus-vm`. There are no ambiguous `wse`/`uswc` suffixes. Changing the region or project name after deployment can replace resources; review the plan.

Select the same subscription in the CLI:

```sh
az account set --subscription YOUR_SUBSCRIPTION_ID
az account show --output table
az vm list-skus --location eastus --resource-type virtualMachines --all --output table
az vm list-usage --location eastus --output table
```

Check subscription policies, regional availability, quota, and pricing before deploying. Azure for Students and paid subscriptions use the same Terraform; their available resources and quotas can differ. See [Azure VM quotas](https://learn.microsoft.com/azure/virtual-machines/quotas) and [Azure for Students](https://learn.microsoft.com/azure/education-hub/about-azure-for-students).

Provider auto-registration is disabled so Terraform does not assume subscription-wide permissions. On your chosen subscription, register these namespaces once if they are not already registered (requires appropriate subscription permissions):

```sh
az provider register --namespace Microsoft.Compute --wait
az provider register --namespace Microsoft.Network --wait
az provider register --namespace Microsoft.DevTestLab --wait
```

## Deploy and connect

```sh
terraform init
terraform fmt -check
terraform validate
terraform plan -var-file=personal.tfvars.json -out=deployment.tfplan
terraform apply deployment.tfplan
terraform output -raw ssh_command
```

The example intentionally cannot deploy until the subscription, SSH key, and SSH source are supplied. State and plans remain local and ignored; keep secure backups of state. Do not commit real subscription settings, state, plans, keys, or command output.

The VM uses key-only authentication and permits inbound SSH from your configured CIDR. To use a notebook later, tunnel its port over SSH rather than opening another public port.

## Lifecycle and costs

Auto-shutdown deallocates the VM each day and does not restart it. Compute stops billing while deallocated, but disks and the static public IP can continue accruing charges. Shutdown also interrupts running experiments; adjust the schedule or temporarily set `auto_shutdown_enabled` to false and apply before a long run.

```sh
az vm start --resource-group transformer-calibration-eastus-rg --name transformer-calibration-eastus-vm
az vm deallocate --resource-group transformer-calibration-eastus-rg --name transformer-calibration-eastus-vm
# Permanently delete this deployment, including its OS disk:
terraform destroy -var-file=personal.tfvars.json
```

Use the resource names from your configuration if you change project or region. Back up experiment data before destroying the VM.

To switch subscriptions, use a separate checkout with its own ignored configuration and independent Terraform state. Do not change the subscription ID against existing state: retain the original checkout/state to manage or destroy that deployment.

## Repository

Remote: https://github.com/MontoyaMMatthew/transformer-calibration-repo

This repository contains the infrastructure scaffold; experiment code and dependencies can be added independently.
