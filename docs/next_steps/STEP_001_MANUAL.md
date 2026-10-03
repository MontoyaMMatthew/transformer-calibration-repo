# Step 001: actions from the Mac

Backup destination selected by the owner:
`/Volumes/X9/projects/transformer-calibration-backups`.

The remote agent can prepare an archive but cannot access the Mac's mounted
external drive. Step 001 remains open until the copy and checksums below are verified.

## 1. Confirm the live shutdown schedule

The Terraform defaults are enabled daily shutdown at 23:00 in the Azure
`Mountain Standard Time` time zone (Denver, including daylight saving).
The VM has no authenticated Azure CLI account, so the current deployed setting
could not be read during setup. In the Azure portal, open
`transformer-calibration-southcentralus-vm` and inspect **Auto-shutdown**.
Record the actual enabled status, time, and time zone in `docs/ENVIRONMENT.md`.
No resize or infrastructure apply is required for CPU setup.

## 2. Copy and verify the setup backup

Run on the **Mac**, using the same SSH host/IP or alias and key options that
you already use to connect to this VM. Replace `YOUR_EXISTING_VM_SSH_HOST`
below with that host. If your connection uses a key selected with `-i`,
add that same option to `scp`.

```bash
mkdir -p /Volumes/X9/projects/transformer-calibration-backups
cd /Volumes/X9/projects/transformer-calibration-backups
scp azureuser@YOUR_EXISTING_VM_SSH_HOST:/home/azureuser/transformer-calibration-repo/artifacts/environment/step-001-setup.tar.gz .
scp azureuser@YOUR_EXISTING_VM_SSH_HOST:/home/azureuser/transformer-calibration-repo/artifacts/environment/step-001-setup.tar.gz.sha256 .
shasum -a 256 -c step-001-setup.tar.gz.sha256
mkdir -p step-001-setup
tar -xzf step-001-setup.tar.gz -C step-001-setup
cd step-001-setup
shasum -a 256 -c SHA256SUMS
```

Both checksum checks must report success. The archive contains setup reports,
resource revisions, scripts, dependency specifications, and documentation.
It excludes `.env`, authentication state, virtual environments, package caches,
model weights, and bulk datasets. The pinned resources can initially be
re-downloaded; an independent weight/data backup is optional.

After verifying, record the backup date and directory in `docs/ENVIRONMENT.md`
and mark the independent-copy checklist items in `STEP_001.md` complete.

## Later GPU work

CPU validation does not verify CUDA, A100 BF16, 32K context, or the resize path.
These belong to the later GPU step. Preserve the DSVM driver installation and
validate the current project environment after resizing through the established
workflow.
