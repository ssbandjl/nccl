/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_INT_DEBUG_H_
#define NCCL_INT_DEBUG_H_

#include "nccl.h"
#include "nccl_common.h"
#include <stdio.h>

#include <pthread.h>

// Conform to pthread and NVTX standard
#define NCCL_THREAD_NAMELEN 16

extern int ncclDebugLevel;
extern FILE *ncclDebugFile;

void ncclDebugLog(ncclDebugLogLevel level, unsigned long flags, const char *filefunc, const char *file, int line, const char *fmt, ...) __attribute__ ((format (printf, 6, 7)));

// Let code temporarily downgrade WARN into INFO
extern thread_local int ncclDebugNoWarn;
extern char ncclLastError[];

#define VERSION(...) ncclDebugLog(NCCL_LOG_VERSION, NCCL_ALL, __func__, __FILE__, __LINE__, __VA_ARGS__)
#define WARN(...) ncclDebugLog(NCCL_LOG_WARN, NCCL_ALL, __func__, __FILE__, __LINE__, __VA_ARGS__)
#define INFO(FLAGS, ...) ncclDebugLog(NCCL_LOG_INFO, (FLAGS), __func__, __FILE__, __LINE__, __VA_ARGS__)
#define TRACE_CALL(...) ncclDebugLog(NCCL_LOG_TRACE, NCCL_CALL, __func__, __FILE__, __LINE__, __VA_ARGS__)

#define NCCL_DEBUG 1
#ifndef printf_ffl
#include <sys/syscall.h>
#define printf_ffl(format, arg...) do {							\
	if (NCCL_DEBUG) {								\
		char hostname[128] = {0};  							\
		gethostname(hostname, sizeof(hostname)); 						\
		printf("[%s] tid:%ld, %s(), %s:%d, " format,					\
			hostname, (long)syscall(SYS_gettid), __FUNCTION__, __FILE__, __LINE__, ##arg);	\
	}										\
} while(0)
#endif

#define NCCL_PRINT_LOG 1
#ifndef printf_log
#include <sys/syscall.h>
#define printf_log(format, arg...) do {							\
	if (NCCL_PRINT_LOG) {								\
		char hostname[128] = {0};  							\
		gethostname(hostname, sizeof(hostname)); 						\
		printf("[%s] tid:%ld, %s(), %s:%d, " format,					\
			hostname, (long)syscall(SYS_gettid), __FUNCTION__, __FILE__, __LINE__, ##arg);	\
	}										\
} while(0)
#endif

#define ENABLE_TRACE
#ifdef ENABLE_TRACE
#define TRACE(FLAGS, ...) ncclDebugLog(NCCL_LOG_TRACE, (FLAGS), __func__, __FILE__, __LINE__, __VA_ARGS__)
#else
#define TRACE(...)
#endif

void ncclSetThreadName(pthread_t thread, const char *fmt, ...);

void ncclResetDebugInit();

#endif
