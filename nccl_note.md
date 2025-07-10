# Nccl Note

# init rdma device
```c
ncclResult_t ncclIbInit(ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction) {
  ncclResult_t ret = ncclSuccess;
  ncclProfilerFunction = profFunction;
  if (ncclParamIbDisable()) return ncclInternalError;
  static int shownIbHcaEnv = 0;
  if(wrap_ibv_symbols() != ncclSuccess) { return ncclInternalError; }

  if (ncclNIbDevs == -1) {
    pthread_mutex_lock(&ncclIbLock);
    wrap_ibv_fork_init();
    if (ncclNIbDevs == -1) {
      ncclNIbDevs = 0;
      ncclNMergedIbDevs = 0;
      if (ncclFindInterfaces(ncclIbIfName, &ncclIbIfAddr, MAX_IF_NAME_SIZE, 1) != 1) {
        WARN("NET/IB : No IP interface found.");
        ret = ncclInternalError;
        goto fail;
      }

      // Detect IB cards
      int nIbDevs;
      struct ibv_device** devices;

      // Check if user defined which IB device:port to use
      const char* userIbEnv = ncclGetEnv("NCCL_IB_HCA");
      if (userIbEnv != NULL && shownIbHcaEnv++ == 0) INFO(NCCL_NET|NCCL_ENV, "NCCL_IB_HCA set to %s", userIbEnv);
      struct netIf userIfs[MAX_IB_DEVS];
      bool searchNot = userIbEnv && userIbEnv[0] == '^';
      if (searchNot) userIbEnv++;
      bool searchExact = userIbEnv && userIbEnv[0] == '=';
      if (searchExact) userIbEnv++;
      int nUserIfs = parseStringList(userIbEnv, userIfs, MAX_IB_DEVS);

      if (ncclSuccess != wrap_ibv_get_device_list(&devices, &nIbDevs)) { ret = ncclInternalError; goto fail; }

      for (int d=0; d<nIbDevs && ncclNIbDevs<MAX_IB_DEVS; d++) {
        struct ibv_context * context;
        dump_stack();
        if (ncclSuccess != wrap_ibv_open_device(&context, devices[d]) || context == NULL) {
          WARN("NET/IB : Unable to open device %s", devices[d]->name);
          continue;
        }
        int nPorts = 0;
        struct ibv_device_attr devAttr;
        memset(&devAttr, 0, sizeof(devAttr));
        if (ncclSuccess != wrap_ibv_query_device(context, &devAttr)) {
          WARN("NET/IB : Unable to query device %s", devices[d]->name);
          if (ncclSuccess != wrap_ibv_close_device(context)) { ret = ncclInternalError; goto fail; }
          continue;
        }
        for (int port_num = 1; port_num <= devAttr.phys_port_cnt; port_num++) {
          struct ibv_port_attr portAttr;
          if (ncclSuccess != wrap_ibv_query_port(context, port_num, &portAttr)) {
            WARN("NET/IB : Unable to query port_num %d", port_num);
            continue;
          }
          if (portAttr.state != IBV_PORT_ACTIVE) continue;
          if (portAttr.link_layer != IBV_LINK_LAYER_INFINIBAND
              && portAttr.link_layer != IBV_LINK_LAYER_ETHERNET) continue;

          // check against user specified HCAs/ports
          if (! (matchIfList(devices[d]->name, port_num, userIfs, nUserIfs, searchExact) ^ searchNot)) {
            continue;
          }
          pthread_mutex_init(&ncclIbDevs[ncclNIbDevs].lock, NULL);
          ncclIbDevs[ncclNIbDevs].device = d;
          ncclIbDevs[ncclNIbDevs].guid = devAttr.sys_image_guid;
          ncclIbDevs[ncclNIbDevs].portAttr = portAttr;
          ncclIbDevs[ncclNIbDevs].portNum = port_num;
          ncclIbDevs[ncclNIbDevs].link = portAttr.link_layer;
          ncclIbDevs[ncclNIbDevs].speed = ncclIbSpeed(portAttr.active_speed) * ncclIbWidth(portAttr.active_width);
          ncclIbDevs[ncclNIbDevs].context = context;
          ncclIbDevs[ncclNIbDevs].pdRefs = 0;
          ncclIbDevs[ncclNIbDevs].pd = NULL;
          strncpy(ncclIbDevs[ncclNIbDevs].devName, devices[d]->name, MAXNAMESIZE);
          NCCLCHECKGOTO(ncclIbGetPciPath(ncclIbDevs[ncclNIbDevs].devName, &ncclIbDevs[ncclNIbDevs].pciPath, &ncclIbDevs[ncclNIbDevs].realPort), ret, fail);
          ncclIbDevs[ncclNIbDevs].maxQp = devAttr.max_qp;
          ncclIbDevs[ncclNIbDevs].mrCache.capacity = 0;
          ncclIbDevs[ncclNIbDevs].mrCache.population = 0;
          ncclIbDevs[ncclNIbDevs].mrCache.slots = NULL;
          NCCLCHECK(ncclIbStatsInit(&ncclIbDevs[ncclNIbDevs].stats));

          // Enable ADAPTIVE_ROUTING by default on IB networks
          // But allow it to be overloaded by an env parameter
          ncclIbDevs[ncclNIbDevs].ar = (portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND) ? 1 : 0;
          if (ncclParamIbAdaptiveRouting() != -2) ncclIbDevs[ncclNIbDevs].ar = ncclParamIbAdaptiveRouting();

          TRACE(NCCL_NET,"NET/IB: [%d] %s:%s:%d/%s speed=%d context=%p pciPath=%s ar=%d", d, devices[d]->name, devices[d]->dev_name, ncclIbDevs[ncclNIbDevs].portNum,
              NCCL_IB_LLSTR(portAttr.link_layer), ncclIbDevs[ncclNIbDevs].speed, context, ncclIbDevs[ncclNIbDevs].pciPath, ncclIbDevs[ncclNIbDevs].ar);

          PTHREADCHECKGOTO(pthread_create(&ncclIbAsyncThread, NULL, ncclIbAsyncThreadMain, ncclIbDevs + ncclNIbDevs), "pthread_create", ret, fail);
          ncclSetThreadName(ncclIbAsyncThread, "NCCL IbAsync %2d", ncclNIbDevs);
          PTHREADCHECKGOTO(pthread_detach(ncclIbAsyncThread), "pthread_detach", ret, fail); // will not be pthread_join()'d

          // Add this plain physical device to the list of virtual devices
          int vDev;
          ncclNetVDeviceProps_t vProps = {0};
          vProps.ndevs = 1;
          vProps.devs[0] = ncclNIbDevs;
          NCCLCHECK(ncclIbMakeVDeviceInternal(&vDev, &vProps));

          ncclNIbDevs++;
          nPorts++;
        }
        if (nPorts == 0 && ncclSuccess != wrap_ibv_close_device(context)) { ret = ncclInternalError; goto fail; }
      }

      if (nIbDevs && (ncclSuccess != wrap_ibv_free_device_list(devices))) { ret = ncclInternalError; goto fail; };
    }
    if (ncclNIbDevs == 0) {
      printf_ffl("NCCL Not find rdma nic device\n");
      INFO(NCCL_INIT|NCCL_NET, "NET/IB : No device found.");
    }

    // Print out all net devices to the user (in the same format as before)
    char line[2048];
    line[0] = '\0';
    // Determine whether RELAXED_ORDERING is enabled and possible
    ncclIbRelaxedOrderingEnabled = ncclIbRelaxedOrderingCapable();
    for (int d = 0; d < ncclNIbDevs; d++) {
        snprintf(line+strlen(line), sizeof(line)-strlen(line), " [%d]%s:%d/%s", d, ncclIbDevs[d].devName,
          ncclIbDevs[d].portNum, NCCL_IB_LLSTR(ncclIbDevs[d].link));
    }
    char addrline[SOCKET_NAME_MAXLEN+1];
    INFO(NCCL_INIT|NCCL_NET, "NET/IB : Using%s %s; OOB %s:%s", line, ncclIbRelaxedOrderingEnabled ? "[RO]" : "",
          ncclIbIfName, ncclSocketToString(&ncclIbIfAddr, addrline));

    pthread_mutex_unlock(&ncclIbLock);
  }
exit:
  return ret;
fail:
  pthread_mutex_unlock(&ncclIbLock);
  goto exit;
}
```


# listen socket
```c
ncclResult_t ncclSocketListen(struct ncclSocket* sock) {
  if (sock == NULL) {
    WARN("ncclSocketListen: pass NULL socket");
    return ncclInvalidArgument;
  }
  if (sock->fd == -1) {
    WARN("ncclSocketListen: file descriptor is -1");
    return ncclInvalidArgument;
  }

  if (socketToPort(&sock->addr)) {
    // Port is forced by env. Make sure we get the port.
    int opt = 1;
    SYSCHECK(setsockopt(sock->fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)), "setsockopt");
#if defined(SO_REUSEPORT)
    SYSCHECK(setsockopt(sock->fd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt)), "setsockopt");
#endif
  }

  // addr port should be 0 (Any port)
  SYSCHECK(bind(sock->fd, &sock->addr.sa, sock->salen), "bind");

  /* Get the assigned Port */
  socklen_t size = sock->salen;
  SYSCHECK(getsockname(sock->fd, &sock->addr.sa, &size), "getsockname");

#ifdef ENABLE_TRACE
  char line[SOCKET_NAME_MAXLEN+1];
  TRACE(NCCL_INIT|NCCL_NET,"Listening on socket %s", ncclSocketToString(&sock->addr, line));
#endif

  /* Put the socket in listen mode
   * NB: The backlog will be silently truncated to the value in /proc/sys/net/core/somaxconn
   */
  SYSCHECK(listen(sock->fd, 16384), "listen");
  sock->state = ncclSocketStateReady;
  return ncclSuccess;
}
```


# bootstrapInit
```c

```


# pytorch use nccl
Great question! Let’s break down **how PyTorch uses NCCL** for multi-GPU and multi-node distributed training.

---

### 🧩 **What is NCCL?**

NCCL (NVIDIA Collective Communication Library) provides **high-performance GPU communication primitives** — like **all-reduce**, **broadcast**, **all-gather** — optimized for:
✅ NVLink,
✅ PCIe,
✅ InfiniBand (RDMA),
✅ RoCE.

PyTorch uses NCCL under the hood to **coordinate tensors between GPUs** efficiently.

---

### 🔥 **How PyTorch integrates NCCL**

---

### ✅ 1️⃣ Single-node multi-GPU

When you run:

```python
import torch
torch.nn.DataParallel(model)
```

→ PyTorch splits batches across multiple GPUs **on one node**, but internally still copies tensors using CUDA device-to-device.

This **does not use NCCL** directly — it relies on single-node CUDA copies.

---

### ✅ 2️⃣ Distributed multi-GPU

For **true distributed parallelism**, you use:

```bash
torchrun --nproc_per_node=4 train.py
```

or

```bash
python -m torch.distributed.launch --nproc_per_node=4 train.py
```

Inside your script:

```python
import torch.distributed as dist
dist.init_process_group(backend='nccl')
```

At this point:
✅ PyTorch uses NCCL to coordinate tensor communication between GPUs,
✅ each GPU runs as a separate process (1 GPU ↔ 1 process),
✅ `DistributedDataParallel` wraps the model for gradient synchronization.

---

### ✅ 3️⃣ Multi-node distributed training

To run across multiple **physical nodes**:

* Set environment variables like:

  ```bash
  MASTER_ADDR=192.168.1.10
  MASTER_PORT=29500
  WORLD_SIZE=8
  RANK=0
  ```
* Launch with:

  ```bash
  torchrun --nnodes=2 --node_rank=0 --nproc_per_node=4 train.py
  ```

Internally, PyTorch:
✅ uses NCCL’s multi-node support,
✅ exchanges `NCCL_UNIQUE_ID` between nodes,
✅ sets up RDMA/InfiniBand (if available) or TCP fallback.

---

### ⚙ **PyTorch NCCL backend flow**

| PyTorch Component                                      | NCCL Usage                                                |
| ------------------------------------------------------ | --------------------------------------------------------- |
| `torch.distributed.init_process_group(backend='nccl')` | Initializes NCCL communicator, exchanges rank info        |
| `DistributedDataParallel`                              | Calls `all_reduce` via NCCL for gradient sync             |
| Multi-node setup                                       | Uses NCCL + `NCCL_SOCKET_IFNAME` for cross-node transport |
| CUDA Streams                                           | NCCL ops run on separate streams for async execution      |

---

### 🏗 **NCCL environment tuning for PyTorch**

| Variable                  | Purpose                                      |
| ------------------------- | -------------------------------------------- |
| `NCCL_DEBUG=INFO`         | Enable NCCL debug logs                       |
| `NCCL_SOCKET_IFNAME=eth0` | Select NIC for cross-node comm               |
| `NCCL_IB_DISABLE=0`       | Enable/disable InfiniBand (default: enabled) |
| `NCCL_P2P_DISABLE=1`      | Disable peer-to-peer comm (for testing)      |

---

### 🔬 **Code example**

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group('nccl')

model = YourModel().cuda()
ddp_model = DDP(model, device_ids=[local_rank])

for data in dataloader:
    output = ddp_model(data)
    loss = compute_loss(output)
    loss.backward()
    optimizer.step()
```

---

### 🚀 **Summary**

✅ PyTorch uses NCCL for:

* Intra-node and inter-node GPU communication,
* Synchronizing gradients in distributed training,
* Maximizing bandwidth on NVLink/PCIe/IB.

✅ You control NCCL behavior through:

* `torch.distributed`,
* environment variables.

---

If you want, I can:
✅ show **detailed NCCL debug logs** from PyTorch,
✅ provide a **ready-to-run multi-node PyTorch script**,
✅ help **tune NCCL performance** on your cluster.

Want me to prepare one for you? Let me know! 🚀





# install nccl
```bash
Network installers (x86)
If not already done, make sure you install the network repository:

Network Installer for Ubuntu24.04

$ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
$ sudo dpkg -i cuda-keyring_1.1-1_all.deb
$ sudo apt-get update
Network Installer for Ubuntu22.04

$ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
$ sudo dpkg -i cuda-keyring_1.1-1_all.deb
$ sudo apt-get update
Network Installer for Ubuntu20.04

$ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
$ sudo dpkg -i cuda-keyring_1.1-1_all.deb
$ sudo apt-get update
Network Installer for RedHat/CentOS 9

$ sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
Network Installer for RedHat/CentOS 8

$ sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
then run the following command to installer NCCL:
For Ubuntu: sudo apt install libnccl2=2.26.5-1+cuda12.9 libnccl-dev=2.26.5-1+cuda12.9
For RHEL/Centos: sudo yum install libnccl-2.26.5-1+cuda12.9 libnccl-devel-2.26.5-1+cuda12.9 libnccl-static-2.26.5-1+cuda12.9
```



