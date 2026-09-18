$pi = "fe80::2ecf:67ff:fe57:9a6f%11"
Write-Host "Waiting for Pi... (reboot the Pi if not up)"
while (-not (Test-Connection -ComputerName $pi -Count 1 -Quiet)) { Start-Sleep -Milliseconds 500 }
Write-Host "Pi UP - enter password"
ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 "miro@$pi" "mkdir -p ~/.ssh; echo 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHp2cXO2VYjNIMEs1EjW3CgD3FVyDBIpVFvaOvtMC6ky ?몄젙??JH_1' >> ~/.ssh/authorized_keys; chmod 700 ~/.ssh; chmod 600 ~/.ssh/authorized_keys; echo KEY_OK"