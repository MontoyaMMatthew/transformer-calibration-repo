locals {
  auto_shutdown_hhmm   = replace(var.auto_shutdown_time, ":", "")
  auto_shutdown_notify = var.auto_shutdown_notification_email != ""
}

resource "azurerm_dev_test_global_vm_shutdown_schedule" "main" {
  virtual_machine_id = azurerm_linux_virtual_machine.main.id
  location           = azurerm_resource_group.main.location
  enabled            = var.auto_shutdown_enabled

  daily_recurrence_time = local.auto_shutdown_hhmm
  timezone              = var.auto_shutdown_timezone

  notification_settings {
    enabled         = local.auto_shutdown_notify
    email           = local.auto_shutdown_notify ? var.auto_shutdown_notification_email : null
    time_in_minutes = 30
  }
}
