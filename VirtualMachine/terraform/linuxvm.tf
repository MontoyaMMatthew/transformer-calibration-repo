resource "azurerm_network_interface" "main" {
  name                = "${local.name_prefix}-nic"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name

  ip_configuration {
    name                          = "primary"
    subnet_id                     = azurerm_subnet.vm-subnet.id
    private_ip_address_allocation = "Dynamic"
    public_ip_address_id          = azurerm_public_ip.main.id
  }
}

resource "azurerm_network_interface_security_group_association" "associate-nsg-nic" {
  network_interface_id      = azurerm_network_interface.main.id
  network_security_group_id = azurerm_network_security_group.main.id
}

resource "azurerm_linux_virtual_machine" "main" {
  name                  = "${local.name_prefix}-vm"
  location              = azurerm_resource_group.main.location
  resource_group_name   = azurerm_resource_group.main.name
  network_interface_ids = [azurerm_network_interface.main.id]

  size = var.vm_size

  disable_password_authentication   = true
  vm_agent_platform_updates_enabled = true

  admin_ssh_key {
    username   = var.admin_username
    public_key = file(pathexpand(var.ssh_public_key_path))
  }

  os_disk {
    name                 = "${local.name_prefix}-osdisk"
    caching              = "ReadWrite"
    storage_account_type = "StandardSSD_LRS"
    disk_size_gb         = var.disk_size_gb
  }

  source_image_reference {
    publisher = "microsoft-dsvm"
    offer     = "ubuntu-2204"
    sku       = "2204-gen2"
    version   = var.dsvm_image_version
  }
  # The verified image has no Marketplace purchase plan. Its GPU stack is
  # preinstalled; do not layer a separate NVIDIA driver extension over it.
  secure_boot_enabled = false
  vtpm_enabled        = false
  computer_name       = "${var.project_name}-vm"
  admin_username      = var.admin_username
}
