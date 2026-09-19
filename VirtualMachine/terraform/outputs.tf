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

output "auto_shutdown" {
  value       = "Your VM deallocates every day at ${substr(local.auto_shutdown_hhmm, 0, 2)}:${substr(local.auto_shutdown_hhmm, 2, 2)} ${var.auto_shutdown_timezone}${var.auto_shutdown_enabled ? "" : " (currently DISABLED by auto_shutdown_enabled)"}. It never starts itself: before each session run 'az vm start --resource-group ${azurerm_resource_group.main.name} --name ${azurerm_linux_virtual_machine.main.name}'."
  description = "When the VM shuts itself down, and how to start it again"
}
