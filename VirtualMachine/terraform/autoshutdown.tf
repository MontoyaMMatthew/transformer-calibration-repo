locals {
  auto_shutdown_hhmm = replace(var.auto_shutdown_time, ":", "")
}

resource "azurerm_dev_test_global_vm_shutdown_schedule" "main" {
  virtual_machine_id = azurerm_linux_virtual_machine.main.id
  location           = azurerm_resource_group.main.location
  enabled            = var.auto_shutdown_enabled

  daily_recurrence_time = local.auto_shutdown_hhmm
  timezone              = var.auto_shutdown_timezone

  notification_settings {
    enabled = false
  }

  lifecycle {
    # Azure returns the target VM ID with different casing on refresh.
    # Ignore that drift, but recreate the schedule when the VM ID changes.
    ignore_changes       = [virtual_machine_id]
    replace_triggered_by = [azurerm_linux_virtual_machine.main.id]
  }
}
