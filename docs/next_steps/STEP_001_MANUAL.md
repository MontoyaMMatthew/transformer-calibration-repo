# Step 001: actions from the Mac

Backup destination selected by the owner:
`/Volumes/X9/projects/transformer-calibration-backups`.

Both manual actions are complete: on October 3, 2026 the owner confirmed the
live daily shutdown schedule at 11pm MST and the backup copy and verification
at the directory above. These are owner confirmations, since the remote agent
cannot access the Mac's mounted external drive. Step 001 is complete. Retain
the instructions below for reference and future backups.

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

The owner confirmation date and backup directory are recorded in
`docs/ENVIRONMENT.md`, and the independent-copy checklist in `STEP_001.md`
is complete.

## Later GPU work

CPU validation does not verify CUDA, A100 BF16, 32K context, or the resize path.
These belong to the later GPU step. Preserve the DSVM driver installation and
validate the current project environment after resizing through the established
workflow.
