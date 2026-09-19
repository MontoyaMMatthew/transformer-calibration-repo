variable "subscription_id" {
  type        = string
  description = "Your personal or Azure for Students subscription ID. Set only in an ignored local tfvars file."
  sensitive   = true
  validation {
    condition     = can(regex("^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$", var.subscription_id))
    error_message = "Replace REDACTED with your own subscription ID in your local configuration."
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
  default     = "eastus"
  description = "Canonical Azure region name, e.g. eastus or westus2. Resource names use this exact region."
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
  default     = "Standard_D2s_v5"
  description = "CPU VM starting size; verify availability and quota for your subscription and region."
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
variable "disk_size_gb" {
  type    = number
  default = 64
  validation {
    condition     = var.disk_size_gb >= 30 && floor(var.disk_size_gb) == var.disk_size_gb
    error_message = "Choose an integer disk size of at least 30 GiB."
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
variable "auto_shutdown_notification_email" {
  type    = string
  default = ""
}
variable "auto_shutdown_enabled" {
  type    = bool
  default = true
}
