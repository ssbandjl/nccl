/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DEBUG_H_
#define NCCL_DEBUG_H_

#include "core.h"

#include <stdio.h>
#include <chrono>

#include <sys/syscall.h>
#include <limits.h>
#include <string.h>
#include "nccl_net.h"

#define gettid() (pid_t) syscall(SYS_gettid)

#define PRINT_LOG (1)

extern int ncclDebugLevel;
extern uint64_t ncclDebugMask;
extern pthread_mutex_t ncclDebugOutputLock;
extern FILE *ncclDebugFile;
extern ncclResult_t getHostName(char* hostname, int maxlen, const char delim);

void ncclDebugLog(ncclDebugLogLevel level, unsigned long flags, const char *filefunc, const char *file, int line, const char *fmt, ...);

// Let code temporarily downgrade WARN into INFO
extern thread_local int ncclDebugNoWarn;
#define NOWARN(a, ret) do { \
  ncclDebugNoWarn = 1; \
  ret = a; \
  ncclDebugNoWarn = 0; \
} while (0)

#define WARN(...) ncclDebugLog(NCCL_LOG_WARN, NCCL_ALL, __func__, __FILE__, __LINE__, __VA_ARGS__)
#define INFO(FLAGS, ...) ncclDebugLog(NCCL_LOG_INFO, (FLAGS), __func__,  __FILE__, __LINE__, __VA_ARGS__)

#ifdef ENABLE_TRACE
#define TRACE(FLAGS, ...) ncclDebugLog(NCCL_LOG_TRACE, (FLAGS), __func__,  __FILE__, __LINE__, __VA_ARGS__)
extern std::chrono::high_resolution_clock::time_point ncclEpoch;
#else
#define TRACE(...)
#endif

#ifndef printf_ffl
#if PRINT_LOG
#define printf_ffl(format, arg...) do { \
  char hostname[1024]; \
  getHostName(hostname, 1024, '.'); \
  printf("[%s] NCCL_XB, %s(), %s:%d, " format, hostname, __func__, __FILE__, __LINE__, ##arg); \
} while (0)
#else
#define printf_ffl(format, arg...) do {} while (0)
#endif
#endif

#endif
