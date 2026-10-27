# System setup: swap and earlyoom (Linux)

Training and image generation need a lot of system RAM, not only GPU memory. sd-scripts
loads the DiT, Qwen3 and the VAE in RAM before moving them to the GPU, and the generator's
low-VRAM profiles keep most DiT blocks (and, in the lightest one, the text encoder) in RAM
on purpose. With a training and the generator running together, a 16 GB machine can run out
of RAM.

When that happens on a Linux desktop with little swap and no out-of-memory daemon, the
kernel does not stop any program: it keeps evicting and reloading memory pages, and the
whole computer freezes (no mouse, no SSH, no remote desktop) until a hard reset. It
happened on the development machine (16 GB RAM, 512 MB swap file, RTX 3060 12 GB) during a
1024x1024 test run with previews.

Two settings prevent it:

1. **Swap of 8-16 GB**, so short peaks fit instead of freezing the system.
2. **earlyoom**, which stops the program using the most memory (the training or the
   generator) a little before RAM and swap run out, and leaves the desktop, SSH and VPN
   running.

Both are system settings that need `sudo`. The app does not change them for you.

## 1. Check what you have

```bash
free -h                       # RAM and swap in use
swapon --show                 # active swap areas
lsblk -f                      # disks and partitions; a swap partition shows FSTYPE "swap"
grep -i swap /etc/fstab       # swap areas enabled at boot
systemctl is-active earlyoom systemd-oomd
```

If `swapon --show` already lists 8 GB or more and `earlyoom` or `systemd-oomd` is
`active`, there is nothing to do.

## 2. Swap

### A swap partition exists but is not used

Installers often create a swap partition and later switch to a swap file, so the partition
stays unused. This is the case when `lsblk -f` shows a `swap` partition that `swapon --show`
does not list. Use its UUID from `lsblk -f` or `sudo blkid`, below shown as `<UUID>`, and
its device, below `/dev/sdXN`:

```bash
sudo swapon /dev/sdXN                                         # use it now
echo 'UUID=<UUID> none swap sw,pri=10 0 0' | sudo tee -a /etc/fstab   # and at every boot
```

### No swap partition: make a larger swap file

```bash
sudo swapoff /swapfile 2>/dev/null; sudo rm -f /swapfile     # old small file, if any
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

On btrfs, a swap file needs `sudo btrfs filesystem mkswapfile --size 16G /swapfile` instead
of `fallocate` and `mkswap`.

### Remove a small swap file you no longer need

Once the partition is active, a small swap file (for example 512 MB) only adds clutter:

```bash
sudo swapoff /swapfile
sudo sed -i 's|^/swapfile|# /swapfile|' /etc/fstab
sudo rm /swapfile
```

### Check

```bash
swapon --show                 # the new swap area, about 8-16 GB
sudo findmnt --verify         # no errors in /etc/fstab before you reboot
```

## 3. earlyoom

If `systemd-oomd` is already active (default on some distributions), keep it and skip
this section: do not run both.

```bash
sudo apt install earlyoom     # Debian, Ubuntu, Kubuntu; Fedora: sudo dnf install earlyoom
sudoedit /etc/default/earlyoom
```

Set the arguments line to:

```
EARLYOOM_ARGS="-r 3600 -m 6 -s 10 --prefer '(^|/)(python3?|accelerate)$' --avoid '(^|/)(plasmashell|kwin_wayland|Xwayland|sddm|gnome-shell|gdm3?|systemd|sshd|tailscaled)$'"
```

- `-m 6 -s 10`: act when 6% or less of RAM **and** 10% or less of swap are free
  (SIGKILL at half those values).
- `--prefer`: stop the training or the generator first (they run as Python processes).
- `--avoid`: never stop the desktop, the login manager, SSH or Tailscale, so the machine
  stays usable and reachable from remote.
- `-r 3600`: write the memory state to the log once an hour.

On Fedora the file is `/etc/sysconfig/earlyoom`. Then enable and restart it:

```bash
sudo systemctl enable --now earlyoom
sudo systemctl restart earlyoom
systemctl status earlyoom --no-pager
```

The status must show the `--prefer` / `--avoid` lines and a `swap total` matching the swap
you set up.

## What you see when earlyoom acts

The training or the generation stops and the app reports the process as killed (for a
training: "killed, probably by the system OOM killer"). The computer keeps working. To
confirm:

```bash
journalctl -u earlyoom --since today --no-pager | grep -i -E 'sending|killed'
```

Then lower the memory use: close other large programs, do not run the generator while a
training runs, or lower the resolution.

## Windows

Windows has no freeze of this kind: it grows the page file and slows down instead. Keep
the page file "managed by the system" and leave at least 16 GB free on its drive (see
milestone M7).
