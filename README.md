# CoPilotIO: CPU as a Co-pilot for GPU I/O to Free GPU Compute

**Guanyi Chen, Qi Chen, Shu Yin, Jian Zhang**

Artifact evaluation for OSDI'26.

**Note:** All commands in this guide should be run as root (`sudo -i` or `sudo su`).

## Hardware Requirements

* A x86 system supporting PCIe P2P.
* A NVMe SSD. Please make sure there isn't any needed data on this SSD as the benchmark writes directly to it.
  * Tested on Samsung PM9A3 7.68TB and Samsung 990 PRO 2TB (both PCIe Gen4 x4).
* A NVIDIA Tesla/Datacenter grade GPU from the Volta or newer generation (compute capability >= 7.0).
  * A Tesla grade GPU is needed as it can expose all of its memory for P2P accesses over PCIe.
  * Tested on A100-SXM4-40GB (108 SMs).
* A system that supports `Above 4G Decoding` for PCIe devices (may need to be ENABLED in the BIOS).

## Software Requirements

* CMake >= 3.1 and the _FindCUDA_ package for CMake.
* GCC >= 5.4.0 with C++11 and POSIX threads support.
* CUDA 12.4 with Nvidia **open-source** driver 550.54.14 (see [below](#install-cuda-and-nvidia-open-source-driver)).
* Linux kernel headers (for building the libnvm and GDRCopy kernel modules).
* Python 3 with `matplotlib` and `numpy` (for plotting).

## System Configuration

### Install CUDA and Nvidia Open-Source Driver

The open-source Nvidia kernel driver is required for the kernel module to build correctly.
If you already have a proprietary Nvidia driver installed, the installer will replace it automatically.

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
bash cuda_12.4.0_550.54.14_linux.run
# Select the nvidia-fs option and choose "open driver"
```

If CUDA toolkit is already installed and you only need to replace the driver (e.g. over SSH):

```bash
bash cuda_12.4.0_550.54.14_linux.run --silent --driver -m=kernel-open --override
```

After installation, add CUDA to your PATH (add to `~/.bashrc` to persist):

```bash
export PATH=/usr/local/cuda/bin:$PATH
```

Then compile the driver kernel module symbols:

```bash
cd /usr/src/nvidia-550.54.14/
make
```

### Disable IOMMU in Linux

IOMMU must be disabled for PCIe peer-to-peer to work correctly.

To check if the IOMMU is on:

```
cat /proc/cmdline | grep iommu
```

If either `iommu=on` or `intel_iommu=on` is found, the IOMMU is enabled.
Disable it by removing these from the `CMDLINE` variable in `/etc/default/grub`
and reconfiguring GRUB. The IOMMU will be disabled after reboot.

On Intel systems, also disable `Vt-d` in the BIOS.
On AMD systems, disable `IOMMU` in the BIOS.
If the option is available, also disable `ACS` in the BIOS.

### Enable 2MB Hugepages

CoPilotIO uses 2MB hugepages for NVMe queue allocation to support large queue depths
on SSDs that require physically contiguous queue memory (NVMe CQR=1).

```bash
# Allocate hugepages (at least 2 per QP, e.g. 512 for 128 QPs)
echo 512 > /proc/sys/vm/nr_hugepages

# Verify
grep HugePages_Total /proc/meminfo
```

To make this persistent across reboots, add to `/etc/sysctl.conf`:

```
vm.nr_hugepages = 512
```

### Set CUDA Module Loading to Eager

CoPilotIO requires eager CUDA module loading to avoid deadlocks when launching
kernels concurrently with persistent GPU threads.

```bash
export CUDA_MODULE_LOADING=EAGER
```

To make this persistent, add the line to `~/.bashrc`.

## Setup

### 0. Install dependencies

```bash
apt update
apt install -y cmake build-essential linux-headers-$(uname -r) python3-matplotlib python3-numpy
```

### 1. Bind NVMe SSD to libnvm

The SSD must be unbound from the default `nvme` driver before loading libnvm.

```bash
# List NVMe devices and their PCI addresses
bash scripts/pci_bind_helpper.sh

# Unbind from nvme driver (replace PCI_ADDR with your device, e.g. 0000:27:00.0)
bash scripts/pci_bind_helpper.sh -u <PCI_ADDR>
```

### 2. Build libnvm library and kernel module (BaM baseline)

```bash
cd bam
mkdir -p build && cd build
cmake ..
make
cd module && make && make load
cd ../../..
```

Verify: `ls /dev/libnvm*` should show `/dev/libnvm0`.

If the device does not appear, check that you unbound the SSD from the `nvme` driver in the previous step.

### 2b. Build CoPilotIO library

```bash
cd copilot-io
mkdir -p build && cd build
cmake ..
make
cd ../..
```

### 3. Build GDRCopy library and load kernel module

```bash
cd gdrcopy
make prefix=gdr_install lib lib_install
cd src/gdrdrv && make
cd ../..
bash insmod.sh
cd ..
```

Verify: `lsmod | grep gdrdrv` should show the module loaded.

### 4. Build the benchmark

```bash
cd microbenchmark
mkdir -p build && cd build
cmake ..
make
cd ../..
```

To rebind the default nvme driver after the whole experiments finished:

```bash
bash scripts/pci_bind_helpper.sh -b <PCI_ADDR>=nvme
```

## Hello World

Quick test to verify everything works:

```bash
cd microbenchmark

LD_LIBRARY_PATH=../copilot-io/build/lib:../gdrcopy/src \
  ./build/nvm-copilotio-read-bw --sms 12 --warps 32 --copilot
```

Should print bandwidth and IOPS results.

For a minimal code example showing CoPilotIO initialization and the synchronous/asynchronous
I/O interfaces, see [`microbenchmark/copilotio_example.cu`](microbenchmark/copilotio_example.cu).

## Pure-I/O Peformance

One command to sweep SM counts for both BaM and CoPilotIO, then generate the plot:

```bash
cd microbenchmark

bash run.sh
```

The script defaults to `/dev/libnvm0`.
If your device mapping differs, edit the `CTRL` variable in `run.sh`.

This runs 4KB random reads at SM counts {12, 24, 48, 96} for both systems,
saves results to `results_bam_read_bw.csv` and `results_copilotio_read_bw.csv`,
and produces `read_bw.pdf` and `read_bw.png`.

## Command-Line Options

```
nvm-bam-read-bw / nvm-copilotio-read-bw [options]
  --ctrl              /dev/libnvmX  (default: /dev/libnvm0)
  --sms               N             (default: 108)
  --warps             N             (default: 32, warps per SM)
  --qps               N             (default: 128)
  --qd                N             (default: 1024)
  --io-size           N             (default: 4096)
  --duration          N             (default: 5, seconds to run)
  --gpu               N             (default: 0)
  --copilot                         enable CoPilotIO mode
  --copilot-base-core N             (default: 16, first CPU core for polling)
  --copilot-cores     N             (default: 0 = one thread per QP)
```

## Loading/Unloading the Kernel Modules

To unload and reload the libnvm kernel module:

```bash
cd bam/build/module
make unload   # unload
make load     # load
make reload   # unload then load
```

To unload GDRCopy:

```bash
rmmod gdrdrv
```

## Contact

* Guanyi Chen — felixlinker02@gmail.com
* Qi Chen - qchen802@connect.hkust-gz.edu.cn
* Jian Zhang — jianz@hkust-gz.edu.cn
