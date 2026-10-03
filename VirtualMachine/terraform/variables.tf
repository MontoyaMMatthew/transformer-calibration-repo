variable "subscription_id" {
  type        = string
  description = "Personal paid subscription ID, shared with the repository's cloud helper through personal.auto.tfvars.json."
  sensitive   = true
  validation {
    condition     = can(regex("^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$", var.subscription_id))
    error_message = "Set your personal subscription ID in personal.auto.tfvars.json."
  }
}
variable "project_name" {
  type    = string
  default = "transformer-calibration"
  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{2,39}$", var.project_name))
    error_message = "Use 3-40 lowercase letters, digits or hyphens, beginning with a letter."
  }
}
variable "location" {
  type        = string
  default     = "southcentralus"
  description = "Deployment region. Changing this after deployment replaces resources."
  validation {
    condition     = can(regex("^[a-z]+[a-z0-9]*$", var.location))
    error_message = "Use a canonical Azure location such as eastus, without spaces."
  }
}
locals {
  name_prefix = "${var.project_name}-${var.location}"
}
variable "admin_username" {
  type    = string
  default = "azureuser"
}
variable "vm_size" {
  type        = string
  default     = "Standard_E8s_v5"
  description = "CPU development or A100 experiments; resizing retains the managed OS disk but restarts the VM."
  validation {
    condition     = contains(["Standard_E8s_v5", "Standard_NC24ads_A100_v4"], var.vm_size)
    error_message = "Choose Standard_E8s_v5 for development or Standard_NC24ads_A100_v4 for experiments."
  }
}
variable "dsvm_image_version" {
  type        = string
  default     = "25.06.18"
  description = "Pinned microsoft-dsvm:ubuntu-2204:2204-gen2 version, verified in South Central US. Changing it replaces the VM."
  validation {
    condition     = can(regex("^[0-9]+\\.[0-9]+\\.[0-9]+$", var.dsvm_image_version))
    error_message = "Pin an explicit DSVM image version; do not use latest."
  }
}
variable "ssh_public_key_path" {
  type    = string
  default = "~/.ssh/transformer_calibration.pub"
  validation {
    condition     = endswith(var.ssh_public_key_path, ".pub") && fileexists(pathexpand(var.ssh_public_key_path))
    error_message = "Provide an existing SSH public key file ending in .pub."
  }
}
variable "allowed_ssh_source" {
  type        = string
  description = "Your public IPv4 address in CIDR notation, normally x.x.x.x/32."
  validation {
    condition     = can(cidrnetmask(var.allowed_ssh_source)) && can(regex("/([1-9]|[12][0-9]|3[0-2])$", var.allowed_ssh_source))
    error_message = "Provide an IPv4 CIDR with prefix 1-32; unrestricted internet access is not accepted."
  }
}
variable "additional_ssh_rules" {
  type = map(object({
    source      = string
    priority    = number
    description = optional(string, "")
  }))
  default     = {}
  description = "Additional named SSH rules for trusted public IPv4 addresses. Priorities must be unique and different from the primary SSH rule's 1001."
  validation {
    condition = alltrue([
      for name, rule in var.additional_ssh_rules :
      name != "SSH" && can(cidrnetmask(rule.source)) &&
      can(regex("/32$", rule.source)) &&
      rule.priority >= 100 && rule.priority <= 4096 &&
      floor(rule.priority) == rule.priority && rule.priority != 1001
    ]) && length(distinct([for rule in values(var.additional_ssh_rules) : rule.priority])) == length(var.additional_ssh_rules)
    error_message = "Use public IPv4 /32 addresses, names other than SSH, and unique integer priorities from 100 to 4096 excluding 1001."
  }
}
variable "disk_size_gb" {
  type    = number
  default = 256
  validation {
    condition     = var.disk_size_gb >= 256 && floor(var.disk_size_gb) == var.disk_size_gb
    error_message = "Choose an integer disk size of at least 256 GiB for the DSVM and research data."
  }
}
variable "auto_shutdown_time" {
  type    = string
  default = "2300"
  validation {
    condition     = can(regex("^([01][0-9]|2[0-3]):?[0-5][0-9]$", var.auto_shutdown_time))
    error_message = "Use HHmm or HH:mm in 24-hour format."
  }
}
variable "auto_shutdown_timezone" {
  type        = string
  default     = "Mountain Standard Time"
  description = "Azure Windows time-zone ID; Mountain Standard Time follows Denver daylight saving."
}
variable "auto_shutdown_enabled" {
  type    = bool
  default = true
}
