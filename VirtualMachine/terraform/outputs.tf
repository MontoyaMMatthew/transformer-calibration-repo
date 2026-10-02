output "public_ip" {
  value       = azurerm_public_ip.main.ip_address
  description = "VM public IP address"
}

output "ssh_command" {
  value       = "ssh -i ${trimsuffix(var.ssh_public_key_path, ".pub")} ${var.admin_username}@${azurerm_public_ip.main.ip_address}"
  description = "Ready-to-paste SSH command"
}

output "resource_group" {
  value       = azurerm_resource_group.main.name
  description = "Resource group holding every resource in this deployment"
}

output "vm_name" {
  value       = azurerm_linux_virtual_machine.main.name
  description = "VM name for start, deallocate, and status commands"
}

output "dsvm_image" {
  value       = "microsoft-dsvm:ubuntu-2204:2204-gen2:${var.dsvm_image_version}"
  description = "Pinned DSVM image used by this deployment"
}
