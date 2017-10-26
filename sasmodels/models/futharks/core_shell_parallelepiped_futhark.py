import sys
import numpy as np
import ctypes as ct
# Stub code for OpenCL setup.

import pyopencl as cl

def parse_preferred_device(s):
    pref_num = 0
    if len(s) > 1 and s[0] == '#':
        i = 1
        while i < len(s):
            if not s[i].isdigit():
                break
            else:
                pref_num = pref_num * 10 + int(s[i])
            i += 1
        while i < len(s) and s[i].isspace():
            i += 1
        return (s[i:], pref_num)
    else:
        return (s, 0)

def get_prefered_context(interactive=False, platform_pref=None, device_pref=None):
    if device_pref != None:
        (device_pref, device_num) = parse_preferred_device(device_pref)
    else:
        device_num = 0

    if interactive:
        return cl.create_some_context(interactive=True)

    def platform_ok(p):
        return not platform_pref or p.name.find(platform_pref) >= 0
    def device_ok(d):
        return not device_pref or d.name.find(device_pref) >= 0

    device_matches = 0

    for p in cl.get_platforms():
        if not platform_ok(p):
            continue
        for d in p.get_devices():
            if not device_ok(d):
                continue
            if device_matches == device_num:
                return cl.Context(devices=[d])
            else:
                device_matches += 1
    raise Exception('No OpenCL platform and device matching constraints found.')
import pyopencl.array
import time
import argparse
FUT_BLOCK_DIM = "16"
synchronous = False
preferred_platform = None
preferred_device = None
fut_opencl_src = """#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void dummy_kernel(__global unsigned char *dummy, int n)
{
    const int thread_gid = get_global_id(0);
    
    if (thread_gid >= n)
        return;
}
typedef char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef long int64_t;
typedef uchar uint8_t;
typedef ushort uint16_t;
typedef uint uint32_t;
typedef ulong uint64_t;
#define ALIGNED_LOCAL_MEMORY(m,size) __local unsigned char m[size] __attribute__ ((align))
static inline int8_t add8(int8_t x, int8_t y)
{
    return x + y;
}
static inline int16_t add16(int16_t x, int16_t y)
{
    return x + y;
}
static inline int32_t add32(int32_t x, int32_t y)
{
    return x + y;
}
static inline int64_t add64(int64_t x, int64_t y)
{
    return x + y;
}
static inline int8_t sub8(int8_t x, int8_t y)
{
    return x - y;
}
static inline int16_t sub16(int16_t x, int16_t y)
{
    return x - y;
}
static inline int32_t sub32(int32_t x, int32_t y)
{
    return x - y;
}
static inline int64_t sub64(int64_t x, int64_t y)
{
    return x - y;
}
static inline int8_t mul8(int8_t x, int8_t y)
{
    return x * y;
}
static inline int16_t mul16(int16_t x, int16_t y)
{
    return x * y;
}
static inline int32_t mul32(int32_t x, int32_t y)
{
    return x * y;
}
static inline int64_t mul64(int64_t x, int64_t y)
{
    return x * y;
}
static inline uint8_t udiv8(uint8_t x, uint8_t y)
{
    return x / y;
}
static inline uint16_t udiv16(uint16_t x, uint16_t y)
{
    return x / y;
}
static inline uint32_t udiv32(uint32_t x, uint32_t y)
{
    return x / y;
}
static inline uint64_t udiv64(uint64_t x, uint64_t y)
{
    return x / y;
}
static inline uint8_t umod8(uint8_t x, uint8_t y)
{
    return x % y;
}
static inline uint16_t umod16(uint16_t x, uint16_t y)
{
    return x % y;
}
static inline uint32_t umod32(uint32_t x, uint32_t y)
{
    return x % y;
}
static inline uint64_t umod64(uint64_t x, uint64_t y)
{
    return x % y;
}
static inline int8_t sdiv8(int8_t x, int8_t y)
{
    int8_t q = x / y;
    int8_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int16_t sdiv16(int16_t x, int16_t y)
{
    int16_t q = x / y;
    int16_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int32_t sdiv32(int32_t x, int32_t y)
{
    int32_t q = x / y;
    int32_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int64_t sdiv64(int64_t x, int64_t y)
{
    int64_t q = x / y;
    int64_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int8_t smod8(int8_t x, int8_t y)
{
    int8_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int16_t smod16(int16_t x, int16_t y)
{
    int16_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int32_t smod32(int32_t x, int32_t y)
{
    int32_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int64_t smod64(int64_t x, int64_t y)
{
    int64_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int8_t squot8(int8_t x, int8_t y)
{
    return x / y;
}
static inline int16_t squot16(int16_t x, int16_t y)
{
    return x / y;
}
static inline int32_t squot32(int32_t x, int32_t y)
{
    return x / y;
}
static inline int64_t squot64(int64_t x, int64_t y)
{
    return x / y;
}
static inline int8_t srem8(int8_t x, int8_t y)
{
    return x % y;
}
static inline int16_t srem16(int16_t x, int16_t y)
{
    return x % y;
}
static inline int32_t srem32(int32_t x, int32_t y)
{
    return x % y;
}
static inline int64_t srem64(int64_t x, int64_t y)
{
    return x % y;
}
static inline int8_t smin8(int8_t x, int8_t y)
{
    return x < y ? x : y;
}
static inline int16_t smin16(int16_t x, int16_t y)
{
    return x < y ? x : y;
}
static inline int32_t smin32(int32_t x, int32_t y)
{
    return x < y ? x : y;
}
static inline int64_t smin64(int64_t x, int64_t y)
{
    return x < y ? x : y;
}
static inline uint8_t umin8(uint8_t x, uint8_t y)
{
    return x < y ? x : y;
}
static inline uint16_t umin16(uint16_t x, uint16_t y)
{
    return x < y ? x : y;
}
static inline uint32_t umin32(uint32_t x, uint32_t y)
{
    return x < y ? x : y;
}
static inline uint64_t umin64(uint64_t x, uint64_t y)
{
    return x < y ? x : y;
}
static inline int8_t smax8(int8_t x, int8_t y)
{
    return x < y ? y : x;
}
static inline int16_t smax16(int16_t x, int16_t y)
{
    return x < y ? y : x;
}
static inline int32_t smax32(int32_t x, int32_t y)
{
    return x < y ? y : x;
}
static inline int64_t smax64(int64_t x, int64_t y)
{
    return x < y ? y : x;
}
static inline uint8_t umax8(uint8_t x, uint8_t y)
{
    return x < y ? y : x;
}
static inline uint16_t umax16(uint16_t x, uint16_t y)
{
    return x < y ? y : x;
}
static inline uint32_t umax32(uint32_t x, uint32_t y)
{
    return x < y ? y : x;
}
static inline uint64_t umax64(uint64_t x, uint64_t y)
{
    return x < y ? y : x;
}
static inline uint8_t shl8(uint8_t x, uint8_t y)
{
    return x << y;
}
static inline uint16_t shl16(uint16_t x, uint16_t y)
{
    return x << y;
}
static inline uint32_t shl32(uint32_t x, uint32_t y)
{
    return x << y;
}
static inline uint64_t shl64(uint64_t x, uint64_t y)
{
    return x << y;
}
static inline uint8_t lshr8(uint8_t x, uint8_t y)
{
    return x >> y;
}
static inline uint16_t lshr16(uint16_t x, uint16_t y)
{
    return x >> y;
}
static inline uint32_t lshr32(uint32_t x, uint32_t y)
{
    return x >> y;
}
static inline uint64_t lshr64(uint64_t x, uint64_t y)
{
    return x >> y;
}
static inline int8_t ashr8(int8_t x, int8_t y)
{
    return x >> y;
}
static inline int16_t ashr16(int16_t x, int16_t y)
{
    return x >> y;
}
static inline int32_t ashr32(int32_t x, int32_t y)
{
    return x >> y;
}
static inline int64_t ashr64(int64_t x, int64_t y)
{
    return x >> y;
}
static inline uint8_t and8(uint8_t x, uint8_t y)
{
    return x & y;
}
static inline uint16_t and16(uint16_t x, uint16_t y)
{
    return x & y;
}
static inline uint32_t and32(uint32_t x, uint32_t y)
{
    return x & y;
}
static inline uint64_t and64(uint64_t x, uint64_t y)
{
    return x & y;
}
static inline uint8_t or8(uint8_t x, uint8_t y)
{
    return x | y;
}
static inline uint16_t or16(uint16_t x, uint16_t y)
{
    return x | y;
}
static inline uint32_t or32(uint32_t x, uint32_t y)
{
    return x | y;
}
static inline uint64_t or64(uint64_t x, uint64_t y)
{
    return x | y;
}
static inline uint8_t xor8(uint8_t x, uint8_t y)
{
    return x ^ y;
}
static inline uint16_t xor16(uint16_t x, uint16_t y)
{
    return x ^ y;
}
static inline uint32_t xor32(uint32_t x, uint32_t y)
{
    return x ^ y;
}
static inline uint64_t xor64(uint64_t x, uint64_t y)
{
    return x ^ y;
}
static inline char ult8(uint8_t x, uint8_t y)
{
    return x < y;
}
static inline char ult16(uint16_t x, uint16_t y)
{
    return x < y;
}
static inline char ult32(uint32_t x, uint32_t y)
{
    return x < y;
}
static inline char ult64(uint64_t x, uint64_t y)
{
    return x < y;
}
static inline char ule8(uint8_t x, uint8_t y)
{
    return x <= y;
}
static inline char ule16(uint16_t x, uint16_t y)
{
    return x <= y;
}
static inline char ule32(uint32_t x, uint32_t y)
{
    return x <= y;
}
static inline char ule64(uint64_t x, uint64_t y)
{
    return x <= y;
}
static inline char slt8(int8_t x, int8_t y)
{
    return x < y;
}
static inline char slt16(int16_t x, int16_t y)
{
    return x < y;
}
static inline char slt32(int32_t x, int32_t y)
{
    return x < y;
}
static inline char slt64(int64_t x, int64_t y)
{
    return x < y;
}
static inline char sle8(int8_t x, int8_t y)
{
    return x <= y;
}
static inline char sle16(int16_t x, int16_t y)
{
    return x <= y;
}
static inline char sle32(int32_t x, int32_t y)
{
    return x <= y;
}
static inline char sle64(int64_t x, int64_t y)
{
    return x <= y;
}
static inline int8_t pow8(int8_t x, int8_t y)
{
    int8_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int16_t pow16(int16_t x, int16_t y)
{
    int16_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int32_t pow32(int32_t x, int32_t y)
{
    int32_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int64_t pow64(int64_t x, int64_t y)
{
    int64_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int8_t sext_i8_i8(int8_t x)
{
    return x;
}
static inline int16_t sext_i8_i16(int8_t x)
{
    return x;
}
static inline int32_t sext_i8_i32(int8_t x)
{
    return x;
}
static inline int64_t sext_i8_i64(int8_t x)
{
    return x;
}
static inline int8_t sext_i16_i8(int16_t x)
{
    return x;
}
static inline int16_t sext_i16_i16(int16_t x)
{
    return x;
}
static inline int32_t sext_i16_i32(int16_t x)
{
    return x;
}
static inline int64_t sext_i16_i64(int16_t x)
{
    return x;
}
static inline int8_t sext_i32_i8(int32_t x)
{
    return x;
}
static inline int16_t sext_i32_i16(int32_t x)
{
    return x;
}
static inline int32_t sext_i32_i32(int32_t x)
{
    return x;
}
static inline int64_t sext_i32_i64(int32_t x)
{
    return x;
}
static inline int8_t sext_i64_i8(int64_t x)
{
    return x;
}
static inline int16_t sext_i64_i16(int64_t x)
{
    return x;
}
static inline int32_t sext_i64_i32(int64_t x)
{
    return x;
}
static inline int64_t sext_i64_i64(int64_t x)
{
    return x;
}
static inline uint8_t zext_i8_i8(uint8_t x)
{
    return x;
}
static inline uint16_t zext_i8_i16(uint8_t x)
{
    return x;
}
static inline uint32_t zext_i8_i32(uint8_t x)
{
    return x;
}
static inline uint64_t zext_i8_i64(uint8_t x)
{
    return x;
}
static inline uint8_t zext_i16_i8(uint16_t x)
{
    return x;
}
static inline uint16_t zext_i16_i16(uint16_t x)
{
    return x;
}
static inline uint32_t zext_i16_i32(uint16_t x)
{
    return x;
}
static inline uint64_t zext_i16_i64(uint16_t x)
{
    return x;
}
static inline uint8_t zext_i32_i8(uint32_t x)
{
    return x;
}
static inline uint16_t zext_i32_i16(uint32_t x)
{
    return x;
}
static inline uint32_t zext_i32_i32(uint32_t x)
{
    return x;
}
static inline uint64_t zext_i32_i64(uint32_t x)
{
    return x;
}
static inline uint8_t zext_i64_i8(uint64_t x)
{
    return x;
}
static inline uint16_t zext_i64_i16(uint64_t x)
{
    return x;
}
static inline uint32_t zext_i64_i32(uint64_t x)
{
    return x;
}
static inline uint64_t zext_i64_i64(uint64_t x)
{
    return x;
}
static inline float fdiv32(float x, float y)
{
    return x / y;
}
static inline float fadd32(float x, float y)
{
    return x + y;
}
static inline float fsub32(float x, float y)
{
    return x - y;
}
static inline float fmul32(float x, float y)
{
    return x * y;
}
static inline float fmin32(float x, float y)
{
    return x < y ? x : y;
}
static inline float fmax32(float x, float y)
{
    return x < y ? y : x;
}
static inline float fpow32(float x, float y)
{
    return pow(x, y);
}
static inline char cmplt32(float x, float y)
{
    return x < y;
}
static inline char cmple32(float x, float y)
{
    return x <= y;
}
static inline float sitofp_i8_f32(int8_t x)
{
    return x;
}
static inline float sitofp_i16_f32(int16_t x)
{
    return x;
}
static inline float sitofp_i32_f32(int32_t x)
{
    return x;
}
static inline float sitofp_i64_f32(int64_t x)
{
    return x;
}
static inline float uitofp_i8_f32(uint8_t x)
{
    return x;
}
static inline float uitofp_i16_f32(uint16_t x)
{
    return x;
}
static inline float uitofp_i32_f32(uint32_t x)
{
    return x;
}
static inline float uitofp_i64_f32(uint64_t x)
{
    return x;
}
static inline int8_t fptosi_f32_i8(float x)
{
    return x;
}
static inline int16_t fptosi_f32_i16(float x)
{
    return x;
}
static inline int32_t fptosi_f32_i32(float x)
{
    return x;
}
static inline int64_t fptosi_f32_i64(float x)
{
    return x;
}
static inline uint8_t fptoui_f32_i8(float x)
{
    return x;
}
static inline uint16_t fptoui_f32_i16(float x)
{
    return x;
}
static inline uint32_t fptoui_f32_i32(float x)
{
    return x;
}
static inline uint64_t fptoui_f32_i64(float x)
{
    return x;
}
static inline double fdiv64(double x, double y)
{
    return x / y;
}
static inline double fadd64(double x, double y)
{
    return x + y;
}
static inline double fsub64(double x, double y)
{
    return x - y;
}
static inline double fmul64(double x, double y)
{
    return x * y;
}
static inline double fmin64(double x, double y)
{
    return x < y ? x : y;
}
static inline double fmax64(double x, double y)
{
    return x < y ? y : x;
}
static inline double fpow64(double x, double y)
{
    return pow(x, y);
}
static inline char cmplt64(double x, double y)
{
    return x < y;
}
static inline char cmple64(double x, double y)
{
    return x <= y;
}
static inline double sitofp_i8_f64(int8_t x)
{
    return x;
}
static inline double sitofp_i16_f64(int16_t x)
{
    return x;
}
static inline double sitofp_i32_f64(int32_t x)
{
    return x;
}
static inline double sitofp_i64_f64(int64_t x)
{
    return x;
}
static inline double uitofp_i8_f64(uint8_t x)
{
    return x;
}
static inline double uitofp_i16_f64(uint16_t x)
{
    return x;
}
static inline double uitofp_i32_f64(uint32_t x)
{
    return x;
}
static inline double uitofp_i64_f64(uint64_t x)
{
    return x;
}
static inline int8_t fptosi_f64_i8(double x)
{
    return x;
}
static inline int16_t fptosi_f64_i16(double x)
{
    return x;
}
static inline int32_t fptosi_f64_i32(double x)
{
    return x;
}
static inline int64_t fptosi_f64_i64(double x)
{
    return x;
}
static inline uint8_t fptoui_f64_i8(double x)
{
    return x;
}
static inline uint16_t fptoui_f64_i16(double x)
{
    return x;
}
static inline uint32_t fptoui_f64_i32(double x)
{
    return x;
}
static inline uint64_t fptoui_f64_i64(double x)
{
    return x;
}
static inline float fpconv_f32_f32(float x)
{
    return x;
}
static inline double fpconv_f32_f64(float x)
{
    return x;
}
static inline float fpconv_f64_f32(double x)
{
    return x;
}
static inline double fpconv_f64_f64(double x)
{
    return x;
}
static inline double futrts_sqrt64(double x)
{
    return sqrt(x);
}
static inline double futrts_sin64(double x)
{
    return sin(x);
}
static inline double futrts_cos64(double x)
{
    return cos(x);
}
#define r_7977 (0.0)
#define group_sizze_9119 (DEFAULT_GROUP_SIZE)
#define r_8662 (0.0)
#define group_sizze_9242 (DEFAULT_GROUP_SIZE)
__kernel void map_kernel_8918(int32_t sizze_7632, double res_7645,
                              double res_7678, double res_7681, double res_7684,
                              double res_7703, double res_7704, double res_7705,
                              double res_7706, double res_7720, double res_7724,
                              double res_7727, double res_7730, double res_7732,
                              double res_7733, double res_7735, double res_7738,
                              double res_7741, double res_7743, double res_7744,
                              double res_7746, double res_7748, double res_7867,
                              __global unsigned char *qx_input_mem_9356,
                              __global unsigned char *qy_input_mem_9358,
                              __global unsigned char *mem_9361)
{
    int32_t wave_sizze_9388;
    int32_t group_sizze_9389;
    char thread_active_9390;
    int32_t gtid_8911;
    int32_t global_tid_8918;
    int32_t local_tid_8919;
    int32_t group_id_8920;
    
    global_tid_8918 = get_global_id(0);
    local_tid_8919 = get_local_id(0);
    group_sizze_9389 = get_local_size(0);
    wave_sizze_9388 = LOCKSTEP_WIDTH;
    group_id_8920 = get_group_id(0);
    gtid_8911 = global_tid_8918;
    thread_active_9390 = slt32(gtid_8911, sizze_7632);
    
    double qx_8921;
    double qy_8922;
    double res_8923;
    double res_8924;
    double res_8925;
    double res_8926;
    double res_8927;
    double res_8928;
    double res_8929;
    double res_8930;
    double res_8931;
    double res_8932;
    double res_8933;
    double res_8934;
    double res_8935;
    double res_8936;
    double res_8937;
    double res_8938;
    double res_8939;
    double res_8940;
    char res_8941;
    double res_8942;
    double res_8945;
    double res_8946;
    char res_8947;
    double res_8948;
    double res_8951;
    double res_8952;
    char res_8953;
    double res_8954;
    double res_8957;
    double res_8958;
    char res_8959;
    double res_8960;
    double res_8963;
    double res_8964;
    char res_8965;
    double res_8966;
    double res_8969;
    double res_8970;
    char res_8971;
    double res_8972;
    double res_8975;
    double res_8976;
    double res_8977;
    double res_8978;
    double res_8979;
    double res_8980;
    double res_8981;
    double res_8982;
    double res_8983;
    double res_8984;
    double res_8985;
    double res_8986;
    double res_8987;
    double res_8988;
    double res_8989;
    double res_8990;
    double res_8991;
    double res_8992;
    double res_8993;
    double res_8994;
    double res_8995;
    double res_8996;
    double res_8997;
    double res_8998;
    double res_8999;
    double res_9000;
    double res_9001;
    
    if (thread_active_9390) {
        qx_8921 = *(__global double *) &qx_input_mem_9356[gtid_8911 * 8];
        qy_8922 = *(__global double *) &qy_input_mem_9358[gtid_8911 * 8];
        res_8923 = qx_8921 * qx_8921;
        res_8924 = qy_8922 * qy_8922;
        res_8925 = res_8923 + res_8924;
        res_8926 = futrts_sqrt64(res_8925);
        res_8927 = qx_8921 / res_8926;
        res_8928 = qy_8922 / res_8926;
        res_8929 = res_8927 * res_7720;
        res_8930 = res_8928 * res_7724;
        res_8931 = res_8929 + res_8930;
        res_8932 = res_8927 * res_7727;
        res_8933 = res_8928 * res_7730;
        res_8934 = res_8932 + res_8933;
        res_8935 = res_8927 * res_7732;
        res_8936 = res_8928 * res_7733;
        res_8937 = res_8935 + res_8936;
        res_8938 = 0.5 * res_8926;
        res_8939 = res_8938 * res_7678;
        res_8940 = res_8939 * res_8931;
        res_8941 = res_8940 == 0.0;
        if (res_8941) {
            res_8942 = 1.0;
        } else {
            double res_8943;
            
            res_8943 = futrts_sin64(res_8940);
            
            double res_8944 = res_8943 / res_8940;
            
            res_8942 = res_8944;
        }
        res_8945 = res_8938 * res_7681;
        res_8946 = res_8945 * res_8934;
        res_8947 = res_8946 == 0.0;
        if (res_8947) {
            res_8948 = 1.0;
        } else {
            double res_8949;
            
            res_8949 = futrts_sin64(res_8946);
            
            double res_8950 = res_8949 / res_8946;
            
            res_8948 = res_8950;
        }
        res_8951 = res_8938 * res_7684;
        res_8952 = res_8951 * res_8937;
        res_8953 = res_8952 == 0.0;
        if (res_8953) {
            res_8954 = 1.0;
        } else {
            double res_8955;
            
            res_8955 = futrts_sin64(res_8952);
            
            double res_8956 = res_8955 / res_8952;
            
            res_8954 = res_8956;
        }
        res_8957 = res_8938 * res_7744;
        res_8958 = res_8957 * res_8931;
        res_8959 = res_8958 == 0.0;
        if (res_8959) {
            res_8960 = 1.0;
        } else {
            double res_8961;
            
            res_8961 = futrts_sin64(res_8958);
            
            double res_8962 = res_8961 / res_8958;
            
            res_8960 = res_8962;
        }
        res_8963 = res_8938 * res_7746;
        res_8964 = res_8963 * res_8934;
        res_8965 = res_8964 == 0.0;
        if (res_8965) {
            res_8966 = 1.0;
        } else {
            double res_8967;
            
            res_8967 = futrts_sin64(res_8964);
            
            double res_8968 = res_8967 / res_8964;
            
            res_8966 = res_8968;
        }
        res_8969 = res_8938 * res_7748;
        res_8970 = res_8969 * res_8937;
        res_8971 = res_8970 == 0.0;
        if (res_8971) {
            res_8972 = 1.0;
        } else {
            double res_8973;
            
            res_8973 = futrts_sin64(res_8970);
            
            double res_8974 = res_8973 / res_8970;
            
            res_8972 = res_8974;
        }
        res_8975 = res_7703 * res_8942;
        res_8976 = res_8975 * res_8948;
        res_8977 = res_8976 * res_8954;
        res_8978 = res_8977 * res_7735;
        res_8979 = res_8960 - res_8942;
        res_8980 = res_7704 * res_8979;
        res_8981 = res_8980 * res_8948;
        res_8982 = res_8981 * res_8954;
        res_8983 = res_8982 * res_7738;
        res_8984 = res_8978 + res_8983;
        res_8985 = res_7705 * res_8942;
        res_8986 = res_8966 - res_8948;
        res_8987 = res_8985 * res_8986;
        res_8988 = res_8987 * res_8954;
        res_8989 = res_8988 * res_7741;
        res_8990 = res_8984 + res_8989;
        res_8991 = res_7706 * res_8942;
        res_8992 = res_8991 * res_8948;
        res_8993 = res_8972 * res_8972;
        res_8994 = res_8993 - res_8954;
        res_8995 = res_8992 * res_8994;
        res_8996 = res_8995 * res_7743;
        res_8997 = res_8990 + res_8996;
        res_8998 = 1.0e-4 * res_8997;
        res_8999 = res_8998 * res_8997;
        res_9000 = res_7867 * res_8999;
        res_9001 = res_9000 + res_7645;
    }
    if (thread_active_9390) {
        *(__global double *) &mem_9361[gtid_8911 * 8] = res_9001;
    }
}
__kernel void map_kernel_9009(int32_t nq_7636, double res_7645, double res_7678,
                              double res_7681, double res_7684, double res_7703,
                              double res_7704, double res_7705, double res_7706,
                              double res_7720, double res_7724, double res_7727,
                              double res_7730, double res_7732, double res_7733,
                              double res_7735, double res_7738, double res_7741,
                              double res_7743, double res_7744, double res_7746,
                              double res_7748, char res_7976, double res_7984,
                              __global unsigned char *qx_input_mem_9356,
                              __global unsigned char *qy_input_mem_9358,
                              __global unsigned char *mem_9364)
{
    int32_t wave_sizze_9391;
    int32_t group_sizze_9392;
    char thread_active_9393;
    int32_t gtid_9002;
    int32_t global_tid_9009;
    int32_t local_tid_9010;
    int32_t group_id_9011;
    
    global_tid_9009 = get_global_id(0);
    local_tid_9010 = get_local_id(0);
    group_sizze_9392 = get_local_size(0);
    wave_sizze_9391 = LOCKSTEP_WIDTH;
    group_id_9011 = get_group_id(0);
    gtid_9002 = global_tid_9009;
    thread_active_9393 = slt32(gtid_9002, nq_7636);
    
    double qx_9012;
    double qy_9013;
    double res_9014;
    double res_9015;
    double res_9016;
    double res_9017;
    double res_9018;
    double res_9019;
    double res_9020;
    double res_9021;
    double res_9022;
    double res_9023;
    double res_9024;
    double res_9025;
    double res_9026;
    double res_9027;
    double res_9028;
    double res_9029;
    double res_9030;
    double res_9031;
    char res_9032;
    double res_9033;
    double res_9036;
    double res_9037;
    char res_9038;
    double res_9039;
    double res_9042;
    double res_9043;
    char res_9044;
    double res_9045;
    double res_9048;
    double res_9049;
    char res_9050;
    double res_9051;
    double res_9054;
    double res_9055;
    char res_9056;
    double res_9057;
    double res_9060;
    double res_9061;
    char res_9062;
    double res_9063;
    double res_9066;
    double res_9067;
    double res_9068;
    double res_9069;
    double res_9070;
    double res_9071;
    double res_9072;
    double res_9073;
    double res_9074;
    double res_9075;
    double res_9076;
    double res_9077;
    double res_9078;
    double res_9079;
    double res_9080;
    double res_9081;
    double res_9082;
    double res_9083;
    double res_9084;
    double res_9085;
    double res_9086;
    double res_9087;
    double res_9088;
    double res_9089;
    double res_9090;
    double res_9091;
    double res_9093;
    double res_9094;
    
    if (thread_active_9393) {
        qx_9012 = *(__global double *) &qx_input_mem_9356[gtid_9002 * 8];
        qy_9013 = *(__global double *) &qy_input_mem_9358[gtid_9002 * 8];
        res_9014 = qx_9012 * qx_9012;
        res_9015 = qy_9013 * qy_9013;
        res_9016 = res_9014 + res_9015;
        res_9017 = futrts_sqrt64(res_9016);
        res_9018 = qx_9012 / res_9017;
        res_9019 = qy_9013 / res_9017;
        res_9020 = res_9018 * res_7720;
        res_9021 = res_9019 * res_7724;
        res_9022 = res_9020 + res_9021;
        res_9023 = res_9018 * res_7727;
        res_9024 = res_9019 * res_7730;
        res_9025 = res_9023 + res_9024;
        res_9026 = res_9018 * res_7732;
        res_9027 = res_9019 * res_7733;
        res_9028 = res_9026 + res_9027;
        res_9029 = 0.5 * res_9017;
        res_9030 = res_9029 * res_7678;
        res_9031 = res_9030 * res_9022;
        res_9032 = res_9031 == 0.0;
        if (res_9032) {
            res_9033 = 1.0;
        } else {
            double res_9034;
            
            res_9034 = futrts_sin64(res_9031);
            
            double res_9035 = res_9034 / res_9031;
            
            res_9033 = res_9035;
        }
        res_9036 = res_9029 * res_7681;
        res_9037 = res_9036 * res_9025;
        res_9038 = res_9037 == 0.0;
        if (res_9038) {
            res_9039 = 1.0;
        } else {
            double res_9040;
            
            res_9040 = futrts_sin64(res_9037);
            
            double res_9041 = res_9040 / res_9037;
            
            res_9039 = res_9041;
        }
        res_9042 = res_9029 * res_7684;
        res_9043 = res_9042 * res_9028;
        res_9044 = res_9043 == 0.0;
        if (res_9044) {
            res_9045 = 1.0;
        } else {
            double res_9046;
            
            res_9046 = futrts_sin64(res_9043);
            
            double res_9047 = res_9046 / res_9043;
            
            res_9045 = res_9047;
        }
        res_9048 = res_9029 * res_7744;
        res_9049 = res_9048 * res_9022;
        res_9050 = res_9049 == 0.0;
        if (res_9050) {
            res_9051 = 1.0;
        } else {
            double res_9052;
            
            res_9052 = futrts_sin64(res_9049);
            
            double res_9053 = res_9052 / res_9049;
            
            res_9051 = res_9053;
        }
        res_9054 = res_9029 * res_7746;
        res_9055 = res_9054 * res_9025;
        res_9056 = res_9055 == 0.0;
        if (res_9056) {
            res_9057 = 1.0;
        } else {
            double res_9058;
            
            res_9058 = futrts_sin64(res_9055);
            
            double res_9059 = res_9058 / res_9055;
            
            res_9057 = res_9059;
        }
        res_9060 = res_9029 * res_7748;
        res_9061 = res_9060 * res_9028;
        res_9062 = res_9061 == 0.0;
        if (res_9062) {
            res_9063 = 1.0;
        } else {
            double res_9064;
            
            res_9064 = futrts_sin64(res_9061);
            
            double res_9065 = res_9064 / res_9061;
            
            res_9063 = res_9065;
        }
        res_9066 = res_7703 * res_9033;
        res_9067 = res_9066 * res_9039;
        res_9068 = res_9067 * res_9045;
        res_9069 = res_9068 * res_7735;
        res_9070 = res_9051 - res_9033;
        res_9071 = res_7704 * res_9070;
        res_9072 = res_9071 * res_9039;
        res_9073 = res_9072 * res_9045;
        res_9074 = res_9073 * res_7738;
        res_9075 = res_9069 + res_9074;
        res_9076 = res_7705 * res_9033;
        res_9077 = res_9057 - res_9039;
        res_9078 = res_9076 * res_9077;
        res_9079 = res_9078 * res_9045;
        res_9080 = res_9079 * res_7741;
        res_9081 = res_9075 + res_9080;
        res_9082 = res_7706 * res_9033;
        res_9083 = res_9082 * res_9039;
        res_9084 = res_9063 * res_9063;
        res_9085 = res_9084 - res_9045;
        res_9086 = res_9083 * res_9085;
        res_9087 = res_9086 * res_7743;
        res_9088 = res_9081 + res_9087;
        res_9089 = 1.0e-4 * res_9088;
        res_9090 = res_9089 * res_9088;
        if (res_7976) {
            double res_9092 = res_9090;
            
            res_9091 = res_9092;
        } else {
            res_9091 = r_7977;
        }
        res_9093 = res_7984 * res_9091;
        res_9094 = res_9093 + res_7645;
    }
    if (thread_active_9393) {
        *(__global double *) &mem_9364[gtid_9002 * 8] = res_9094;
    }
}
__kernel void map_kernel_9124(__local volatile int64_t *mem_aligned_0,
                              __local volatile int64_t *mem_aligned_1,
                              int32_t sizze_8085, double res_8096,
                              double res_8126, double res_8136, double res_8137,
                              double res_8140, double res_8143, double res_8154,
                              double res_8155, double res_8158, double res_8416,
                              __global unsigned char *q_input_mem_9356, __global
                              unsigned char *mem_9359, __global
                              unsigned char *mem_9362, __global
                              unsigned char *mem_9371)
{
    __local volatile char *restrict mem_9365 = mem_aligned_0;
    __local volatile char *restrict mem_9368 = mem_aligned_1;
    int32_t wave_sizze_9399;
    int32_t group_sizze_9400;
    char thread_active_9401;
    int32_t gtid_9117;
    int32_t global_tid_9124;
    int32_t local_tid_9125;
    int32_t group_id_9126;
    
    global_tid_9124 = get_global_id(0);
    local_tid_9125 = get_local_id(0);
    group_sizze_9400 = get_local_size(0);
    wave_sizze_9399 = LOCKSTEP_WIDTH;
    group_id_9126 = get_group_id(0);
    gtid_9117 = global_tid_9124;
    thread_active_9401 = slt32(gtid_9117, sizze_8085);
    
    double q_9127;
    double res_9128;
    double res_9129;
    double res_9130;
    
    if (thread_active_9401) {
        q_9127 = *(__global double *) &q_input_mem_9356[gtid_9117 * 8];
        res_9128 = 0.5 * q_9127;
        res_9129 = res_9128 * res_8126;
        res_9130 = res_9129 * res_8137;
    }
    
    double arg_9131;
    double binop_param_x_9134 = 0.0;
    int32_t chunk_sizze_9132;
    int32_t chunk_offset_9133 = 0;
    
    while (slt32(chunk_offset_9133, 76)) {
        if (slt32(76 - chunk_offset_9133, group_sizze_9119)) {
            chunk_sizze_9132 = 76 - chunk_offset_9133;
        } else {
            chunk_sizze_9132 = group_sizze_9119;
        }
        
        double arg_9137;
        
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt32(local_tid_9125, chunk_sizze_9132) && 1) {
            double gaussZZ_chunk_outer_elem_9344 = *(__global
                                                     double *) &mem_9359[(chunk_offset_9133 +
                                                                          local_tid_9125) *
                                                                         8];
            
            *(__local double *) &mem_9365[local_tid_9125 * 8] =
                gaussZZ_chunk_outer_elem_9344;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt32(local_tid_9125, chunk_sizze_9132) && 1) {
            double gaussWt_chunk_outer_elem_9346 = *(__global
                                                     double *) &mem_9362[(chunk_offset_9133 +
                                                                          local_tid_9125) *
                                                                         8];
            
            *(__local double *) &mem_9368[local_tid_9125 * 8] =
                gaussWt_chunk_outer_elem_9346;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        double acc_9140 = binop_param_x_9134;
        int32_t groupstream_mapaccum_dummy_chunk_sizze_9138;
        int32_t i_9139 = 0;
        
        groupstream_mapaccum_dummy_chunk_sizze_9138 = chunk_sizze_9132;
        for (int32_t i_9139 = 0; i_9139 < chunk_sizze_9132; i_9139++) {
            double gaussZZ_9143;
            double gaussWt_9144;
            double res_9146;
            double res_9147;
            double res_9148;
            double res_9149;
            double res_9150;
            double res_9151;
            
            if (thread_active_9401) {
                gaussZZ_9143 = *(__local double *) &mem_9365[i_9139 * 8];
                gaussWt_9144 = *(__local double *) &mem_9368[i_9139 * 8];
                res_9146 = gaussZZ_9143 + 1.0;
                res_9147 = 0.5 * res_9146;
                res_9148 = fpow64(res_9147, 2.0);
                res_9149 = 1.0 - res_9148;
                res_9150 = futrts_sqrt64(res_9149);
                res_9151 = res_9129 * res_9150;
            }
            
            double arg_9152;
            double binop_param_x_9155 = 0.0;
            int32_t chunk_sizze_9153;
            int32_t chunk_offset_9154 = 0;
            
            chunk_sizze_9153 = 76;
            
            double arg_9158;
            double acc_9161 = binop_param_x_9155;
            int32_t groupstream_mapaccum_dummy_chunk_sizze_9159 = 1;
            
            if (thread_active_9401) {
                if (chunk_sizze_9153 == 76) {
                    for (int32_t i_9160 = 0; i_9160 < 76; i_9160++) {
                        double gaussZZ_9164 = *(__global
                                                double *) &mem_9359[(chunk_offset_9154 +
                                                                     i_9160) *
                                                                    8];
                        double gaussWt_9165 = *(__global
                                                double *) &mem_9362[(chunk_offset_9154 +
                                                                     i_9160) *
                                                                    8];
                        double res_9167 = gaussZZ_9164 + 1.0;
                        double res_9168 = 0.5 * res_9167;
                        double res_9169 = 1.570796326794897 * res_9168;
                        double res_9170;
                        
                        res_9170 = futrts_sin64(res_9169);
                        
                        double res_9171;
                        
                        res_9171 = futrts_cos64(res_9169);
                        
                        double res_9172 = res_9151 * res_9170;
                        double res_9173 = res_9172 * res_8136;
                        char res_9174 = res_9173 == 0.0;
                        double res_9175;
                        
                        if (res_9174) {
                            res_9175 = 1.0;
                        } else {
                            double res_9176;
                            
                            res_9176 = futrts_sin64(res_9173);
                            
                            double res_9177 = res_9176 / res_9173;
                            
                            res_9175 = res_9177;
                        }
                        
                        double res_9178 = res_9151 * res_9171;
                        char res_9179 = res_9178 == 0.0;
                        double res_9180;
                        
                        if (res_9179) {
                            res_9180 = 1.0;
                        } else {
                            double res_9181;
                            
                            res_9181 = futrts_sin64(res_9178);
                            
                            double res_9182 = res_9181 / res_9178;
                            
                            res_9180 = res_9182;
                        }
                        
                        double res_9183 = res_9172 * res_8140;
                        char res_9184 = res_9183 == 0.0;
                        double res_9185;
                        
                        if (res_9184) {
                            res_9185 = 1.0;
                        } else {
                            double res_9186;
                            
                            res_9186 = futrts_sin64(res_9183);
                            
                            double res_9187 = res_9186 / res_9183;
                            
                            res_9185 = res_9187;
                        }
                        
                        double res_9188 = res_9178 * res_8143;
                        char res_9189 = res_9188 == 0.0;
                        double res_9190;
                        
                        if (res_9189) {
                            res_9190 = 1.0;
                        } else {
                            double res_9191;
                            
                            res_9191 = futrts_sin64(res_9188);
                            
                            double res_9192 = res_9191 / res_9188;
                            
                            res_9190 = res_9192;
                        }
                        
                        double res_9193 = res_8158 * res_9175;
                        double res_9194 = res_9193 * res_9180;
                        double res_9195 = res_8154 * res_9180;
                        double res_9196 = res_9195 * res_9185;
                        double res_9197 = res_9194 + res_9196;
                        double res_9198 = res_8155 * res_9175;
                        double res_9199 = res_9198 * res_9190;
                        double res_9200 = res_9197 + res_9199;
                        double res_9201 = gaussWt_9165 * res_9200;
                        double res_9202 = res_9201 * res_9200;
                        double res_9203 = acc_9161 + res_9202;
                        
                        acc_9161 = res_9203;
                    }
                } else {
                    for (int32_t i_9160 = 0; i_9160 < chunk_sizze_9153;
                         i_9160++) {
                        double gaussZZ_9164 = *(__global
                                                double *) &mem_9359[(chunk_offset_9154 +
                                                                     i_9160) *
                                                                    8];
                        double gaussWt_9165 = *(__global
                                                double *) &mem_9362[(chunk_offset_9154 +
                                                                     i_9160) *
                                                                    8];
                        double res_9167 = gaussZZ_9164 + 1.0;
                        double res_9168 = 0.5 * res_9167;
                        double res_9169 = 1.570796326794897 * res_9168;
                        double res_9170;
                        
                        res_9170 = futrts_sin64(res_9169);
                        
                        double res_9171;
                        
                        res_9171 = futrts_cos64(res_9169);
                        
                        double res_9172 = res_9151 * res_9170;
                        double res_9173 = res_9172 * res_8136;
                        char res_9174 = res_9173 == 0.0;
                        double res_9175;
                        
                        if (res_9174) {
                            res_9175 = 1.0;
                        } else {
                            double res_9176;
                            
                            res_9176 = futrts_sin64(res_9173);
                            
                            double res_9177 = res_9176 / res_9173;
                            
                            res_9175 = res_9177;
                        }
                        
                        double res_9178 = res_9151 * res_9171;
                        char res_9179 = res_9178 == 0.0;
                        double res_9180;
                        
                        if (res_9179) {
                            res_9180 = 1.0;
                        } else {
                            double res_9181;
                            
                            res_9181 = futrts_sin64(res_9178);
                            
                            double res_9182 = res_9181 / res_9178;
                            
                            res_9180 = res_9182;
                        }
                        
                        double res_9183 = res_9172 * res_8140;
                        char res_9184 = res_9183 == 0.0;
                        double res_9185;
                        
                        if (res_9184) {
                            res_9185 = 1.0;
                        } else {
                            double res_9186;
                            
                            res_9186 = futrts_sin64(res_9183);
                            
                            double res_9187 = res_9186 / res_9183;
                            
                            res_9185 = res_9187;
                        }
                        
                        double res_9188 = res_9178 * res_8143;
                        char res_9189 = res_9188 == 0.0;
                        double res_9190;
                        
                        if (res_9189) {
                            res_9190 = 1.0;
                        } else {
                            double res_9191;
                            
                            res_9191 = futrts_sin64(res_9188);
                            
                            double res_9192 = res_9191 / res_9188;
                            
                            res_9190 = res_9192;
                        }
                        
                        double res_9193 = res_8158 * res_9175;
                        double res_9194 = res_9193 * res_9180;
                        double res_9195 = res_8154 * res_9180;
                        double res_9196 = res_9195 * res_9185;
                        double res_9197 = res_9194 + res_9196;
                        double res_9198 = res_8155 * res_9175;
                        double res_9199 = res_9198 * res_9190;
                        double res_9200 = res_9197 + res_9199;
                        double res_9201 = gaussWt_9165 * res_9200;
                        double res_9202 = res_9201 * res_9200;
                        double res_9203 = acc_9161 + res_9202;
                        
                        acc_9161 = res_9203;
                    }
                }
            }
            arg_9158 = acc_9161;
            binop_param_x_9155 = arg_9158;
            arg_9152 = binop_param_x_9155;
            
            double res_9204;
            double res_9205;
            char res_9206;
            double res_9207;
            double res_9210;
            double res_9211;
            double res_9212;
            double res_9213;
            
            if (thread_active_9401) {
                res_9204 = arg_9152 / 2.0;
                res_9205 = res_9130 * res_9147;
                res_9206 = res_9205 == 0.0;
                if (res_9206) {
                    res_9207 = 1.0;
                } else {
                    double res_9208;
                    
                    res_9208 = futrts_sin64(res_9205);
                    
                    double res_9209 = res_9208 / res_9205;
                    
                    res_9207 = res_9209;
                }
                res_9210 = gaussWt_9144 * res_9204;
                res_9211 = res_9210 * res_9207;
                res_9212 = res_9211 * res_9207;
                res_9213 = acc_9140 + res_9212;
            }
            acc_9140 = res_9213;
        }
        arg_9137 = acc_9140;
        binop_param_x_9134 = arg_9137;
        chunk_offset_9133 += group_sizze_9119;
    }
    arg_9131 = binop_param_x_9134;
    
    double res_9214;
    double res_9215;
    double res_9216;
    double res_9217;
    
    if (thread_active_9401) {
        res_9214 = arg_9131 / 2.0;
        res_9215 = 1.0e-4 * res_9214;
        res_9216 = res_8416 * res_9215;
        res_9217 = res_9216 + res_8096;
    }
    if (thread_active_9401) {
        *(__global double *) &mem_9371[gtid_9117 * 8] = res_9217;
    }
}
__kernel void map_kernel_9247(__local volatile int64_t *mem_aligned_0,
                              __local volatile int64_t *mem_aligned_1,
                              int32_t nq_8088, double res_8096, double res_8126,
                              double res_8136, double res_8137, double res_8140,
                              double res_8143, double res_8154, double res_8155,
                              double res_8158, char res_8661, double res_8669,
                              __global unsigned char *q_input_mem_9356, __global
                              unsigned char *mem_9359, __global
                              unsigned char *mem_9362, __global
                              unsigned char *mem_9380)
{
    __local volatile char *restrict mem_9374 = mem_aligned_0;
    __local volatile char *restrict mem_9377 = mem_aligned_1;
    int32_t wave_sizze_9402;
    int32_t group_sizze_9403;
    char thread_active_9404;
    int32_t gtid_9240;
    int32_t global_tid_9247;
    int32_t local_tid_9248;
    int32_t group_id_9249;
    
    global_tid_9247 = get_global_id(0);
    local_tid_9248 = get_local_id(0);
    group_sizze_9403 = get_local_size(0);
    wave_sizze_9402 = LOCKSTEP_WIDTH;
    group_id_9249 = get_group_id(0);
    gtid_9240 = global_tid_9247;
    thread_active_9404 = slt32(gtid_9240, nq_8088);
    
    double q_9250;
    double res_9251;
    double res_9252;
    double res_9253;
    
    if (thread_active_9404) {
        q_9250 = *(__global double *) &q_input_mem_9356[gtid_9240 * 8];
        res_9251 = 0.5 * q_9250;
        res_9252 = res_9251 * res_8126;
        res_9253 = res_9252 * res_8137;
    }
    
    double arg_9254;
    double binop_param_x_9257 = 0.0;
    int32_t chunk_sizze_9255;
    int32_t chunk_offset_9256 = 0;
    
    while (slt32(chunk_offset_9256, 76)) {
        if (slt32(76 - chunk_offset_9256, group_sizze_9242)) {
            chunk_sizze_9255 = 76 - chunk_offset_9256;
        } else {
            chunk_sizze_9255 = group_sizze_9242;
        }
        
        double arg_9260;
        
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt32(local_tid_9248, chunk_sizze_9255) && 1) {
            double gaussZZ_chunk_outer_elem_9348 = *(__global
                                                     double *) &mem_9359[(chunk_offset_9256 +
                                                                          local_tid_9248) *
                                                                         8];
            
            *(__local double *) &mem_9374[local_tid_9248 * 8] =
                gaussZZ_chunk_outer_elem_9348;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt32(local_tid_9248, chunk_sizze_9255) && 1) {
            double gaussWt_chunk_outer_elem_9350 = *(__global
                                                     double *) &mem_9362[(chunk_offset_9256 +
                                                                          local_tid_9248) *
                                                                         8];
            
            *(__local double *) &mem_9377[local_tid_9248 * 8] =
                gaussWt_chunk_outer_elem_9350;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        double acc_9263 = binop_param_x_9257;
        int32_t groupstream_mapaccum_dummy_chunk_sizze_9261;
        int32_t i_9262 = 0;
        
        groupstream_mapaccum_dummy_chunk_sizze_9261 = chunk_sizze_9255;
        for (int32_t i_9262 = 0; i_9262 < chunk_sizze_9255; i_9262++) {
            double gaussZZ_9266;
            double gaussWt_9267;
            double res_9269;
            double res_9270;
            double res_9271;
            double res_9272;
            double res_9273;
            double res_9274;
            
            if (thread_active_9404) {
                gaussZZ_9266 = *(__local double *) &mem_9374[i_9262 * 8];
                gaussWt_9267 = *(__local double *) &mem_9377[i_9262 * 8];
                res_9269 = gaussZZ_9266 + 1.0;
                res_9270 = 0.5 * res_9269;
                res_9271 = fpow64(res_9270, 2.0);
                res_9272 = 1.0 - res_9271;
                res_9273 = futrts_sqrt64(res_9272);
                res_9274 = res_9252 * res_9273;
            }
            
            double arg_9275;
            double binop_param_x_9278 = 0.0;
            int32_t chunk_sizze_9276;
            int32_t chunk_offset_9277 = 0;
            
            chunk_sizze_9276 = 76;
            
            double arg_9281;
            double acc_9284 = binop_param_x_9278;
            int32_t groupstream_mapaccum_dummy_chunk_sizze_9282 = 1;
            
            if (thread_active_9404) {
                if (chunk_sizze_9276 == 76) {
                    for (int32_t i_9283 = 0; i_9283 < 76; i_9283++) {
                        double gaussZZ_9287 = *(__global
                                                double *) &mem_9359[(chunk_offset_9277 +
                                                                     i_9283) *
                                                                    8];
                        double gaussWt_9288 = *(__global
                                                double *) &mem_9362[(chunk_offset_9277 +
                                                                     i_9283) *
                                                                    8];
                        double res_9290 = gaussZZ_9287 + 1.0;
                        double res_9291 = 0.5 * res_9290;
                        double res_9292 = 1.570796326794897 * res_9291;
                        double res_9293;
                        
                        res_9293 = futrts_sin64(res_9292);
                        
                        double res_9294;
                        
                        res_9294 = futrts_cos64(res_9292);
                        
                        double res_9295 = res_9274 * res_9293;
                        double res_9296 = res_9295 * res_8136;
                        char res_9297 = res_9296 == 0.0;
                        double res_9298;
                        
                        if (res_9297) {
                            res_9298 = 1.0;
                        } else {
                            double res_9299;
                            
                            res_9299 = futrts_sin64(res_9296);
                            
                            double res_9300 = res_9299 / res_9296;
                            
                            res_9298 = res_9300;
                        }
                        
                        double res_9301 = res_9274 * res_9294;
                        char res_9302 = res_9301 == 0.0;
                        double res_9303;
                        
                        if (res_9302) {
                            res_9303 = 1.0;
                        } else {
                            double res_9304;
                            
                            res_9304 = futrts_sin64(res_9301);
                            
                            double res_9305 = res_9304 / res_9301;
                            
                            res_9303 = res_9305;
                        }
                        
                        double res_9306 = res_9295 * res_8140;
                        char res_9307 = res_9306 == 0.0;
                        double res_9308;
                        
                        if (res_9307) {
                            res_9308 = 1.0;
                        } else {
                            double res_9309;
                            
                            res_9309 = futrts_sin64(res_9306);
                            
                            double res_9310 = res_9309 / res_9306;
                            
                            res_9308 = res_9310;
                        }
                        
                        double res_9311 = res_9301 * res_8143;
                        char res_9312 = res_9311 == 0.0;
                        double res_9313;
                        
                        if (res_9312) {
                            res_9313 = 1.0;
                        } else {
                            double res_9314;
                            
                            res_9314 = futrts_sin64(res_9311);
                            
                            double res_9315 = res_9314 / res_9311;
                            
                            res_9313 = res_9315;
                        }
                        
                        double res_9316 = res_8158 * res_9298;
                        double res_9317 = res_9316 * res_9303;
                        double res_9318 = res_8154 * res_9303;
                        double res_9319 = res_9318 * res_9308;
                        double res_9320 = res_9317 + res_9319;
                        double res_9321 = res_8155 * res_9298;
                        double res_9322 = res_9321 * res_9313;
                        double res_9323 = res_9320 + res_9322;
                        double res_9324 = gaussWt_9288 * res_9323;
                        double res_9325 = res_9324 * res_9323;
                        double res_9326 = acc_9284 + res_9325;
                        
                        acc_9284 = res_9326;
                    }
                } else {
                    for (int32_t i_9283 = 0; i_9283 < chunk_sizze_9276;
                         i_9283++) {
                        double gaussZZ_9287 = *(__global
                                                double *) &mem_9359[(chunk_offset_9277 +
                                                                     i_9283) *
                                                                    8];
                        double gaussWt_9288 = *(__global
                                                double *) &mem_9362[(chunk_offset_9277 +
                                                                     i_9283) *
                                                                    8];
                        double res_9290 = gaussZZ_9287 + 1.0;
                        double res_9291 = 0.5 * res_9290;
                        double res_9292 = 1.570796326794897 * res_9291;
                        double res_9293;
                        
                        res_9293 = futrts_sin64(res_9292);
                        
                        double res_9294;
                        
                        res_9294 = futrts_cos64(res_9292);
                        
                        double res_9295 = res_9274 * res_9293;
                        double res_9296 = res_9295 * res_8136;
                        char res_9297 = res_9296 == 0.0;
                        double res_9298;
                        
                        if (res_9297) {
                            res_9298 = 1.0;
                        } else {
                            double res_9299;
                            
                            res_9299 = futrts_sin64(res_9296);
                            
                            double res_9300 = res_9299 / res_9296;
                            
                            res_9298 = res_9300;
                        }
                        
                        double res_9301 = res_9274 * res_9294;
                        char res_9302 = res_9301 == 0.0;
                        double res_9303;
                        
                        if (res_9302) {
                            res_9303 = 1.0;
                        } else {
                            double res_9304;
                            
                            res_9304 = futrts_sin64(res_9301);
                            
                            double res_9305 = res_9304 / res_9301;
                            
                            res_9303 = res_9305;
                        }
                        
                        double res_9306 = res_9295 * res_8140;
                        char res_9307 = res_9306 == 0.0;
                        double res_9308;
                        
                        if (res_9307) {
                            res_9308 = 1.0;
                        } else {
                            double res_9309;
                            
                            res_9309 = futrts_sin64(res_9306);
                            
                            double res_9310 = res_9309 / res_9306;
                            
                            res_9308 = res_9310;
                        }
                        
                        double res_9311 = res_9301 * res_8143;
                        char res_9312 = res_9311 == 0.0;
                        double res_9313;
                        
                        if (res_9312) {
                            res_9313 = 1.0;
                        } else {
                            double res_9314;
                            
                            res_9314 = futrts_sin64(res_9311);
                            
                            double res_9315 = res_9314 / res_9311;
                            
                            res_9313 = res_9315;
                        }
                        
                        double res_9316 = res_8158 * res_9298;
                        double res_9317 = res_9316 * res_9303;
                        double res_9318 = res_8154 * res_9303;
                        double res_9319 = res_9318 * res_9308;
                        double res_9320 = res_9317 + res_9319;
                        double res_9321 = res_8155 * res_9298;
                        double res_9322 = res_9321 * res_9313;
                        double res_9323 = res_9320 + res_9322;
                        double res_9324 = gaussWt_9288 * res_9323;
                        double res_9325 = res_9324 * res_9323;
                        double res_9326 = acc_9284 + res_9325;
                        
                        acc_9284 = res_9326;
                    }
                }
            }
            arg_9281 = acc_9284;
            binop_param_x_9278 = arg_9281;
            arg_9275 = binop_param_x_9278;
            
            double res_9327;
            double res_9328;
            char res_9329;
            double res_9330;
            double res_9333;
            double res_9334;
            double res_9335;
            double res_9336;
            
            if (thread_active_9404) {
                res_9327 = arg_9275 / 2.0;
                res_9328 = res_9253 * res_9270;
                res_9329 = res_9328 == 0.0;
                if (res_9329) {
                    res_9330 = 1.0;
                } else {
                    double res_9331;
                    
                    res_9331 = futrts_sin64(res_9328);
                    
                    double res_9332 = res_9331 / res_9328;
                    
                    res_9330 = res_9332;
                }
                res_9333 = gaussWt_9267 * res_9327;
                res_9334 = res_9333 * res_9330;
                res_9335 = res_9334 * res_9330;
                res_9336 = acc_9263 + res_9335;
            }
            acc_9263 = res_9336;
        }
        arg_9260 = acc_9263;
        binop_param_x_9257 = arg_9260;
        chunk_offset_9256 += group_sizze_9242;
    }
    arg_9254 = binop_param_x_9257;
    
    double res_9337;
    double res_9338;
    double res_9339;
    double res_9341;
    double res_9342;
    
    if (thread_active_9404) {
        res_9337 = arg_9254 / 2.0;
        res_9338 = 1.0e-4 * res_9337;
        if (res_8661) {
            double res_9340 = res_9338;
            
            res_9339 = res_9340;
        } else {
            res_9339 = r_8662;
        }
        res_9341 = res_8669 * res_9339;
        res_9342 = res_9341 + res_8096;
    }
    if (thread_active_9404) {
        *(__global double *) &mem_9380[gtid_9240 * 8] = res_9342;
    }
}
"""
# Hacky parser/reader for values written in Futhark syntax.  Used for
# reading stdin when compiling standalone programs with the Python
# code generator.

import numpy as np
import string
import struct
import sys

lookahead_buffer = []

def reset_lookahead():
    global lookahead_buffer
    lookahead_buffer = []

def get_char(f):
    global lookahead_buffer
    if len(lookahead_buffer) == 0:
        return f.read(1)
    else:
        c = lookahead_buffer[0]
        lookahead_buffer = lookahead_buffer[1:]
        return c

def get_chars(f, n):
    s = b''
    for _ in range(n):
        s += get_char(f)
    return s

def unget_char(f, c):
    global lookahead_buffer
    lookahead_buffer = [c] + lookahead_buffer

def peek_char(f):
    c = get_char(f)
    if c:
        unget_char(f, c)
    return c

def skip_spaces(f):
    c = get_char(f)
    while c != None:
        if c.isspace():
            c = get_char(f)
        elif c == b'-':
          # May be line comment.
          if peek_char(f) == b'-':
            # Yes, line comment. Skip to end of line.
            while (c != b'\n' and c != None):
              c = get_char(f)
          else:
            break
        else:
          break
    if c:
        unget_char(f, c)

def parse_specific_char(f, expected):
    got = get_char(f)
    if got != expected:
        unget_char(f, got)
        raise ValueError
    return True

def parse_specific_string(f, s):
    # This funky mess is intended, and is caused by the fact that if `type(b) ==
    # bytes` then `type(b[0]) == int`, but we need to match each element with a
    # `bytes`, so therefore we make each character an array element
    b = s.encode('utf8')
    bs = [b[i:i+1] for i in range(len(b))]
    for c in bs:
        parse_specific_char(f, c)
    return True

def optional(p, *args):
    try:
        return p(*args)
    except ValueError:
        return None

def optional_specific_string(f, s):
    c = peek_char(f)
    # This funky mess is intended, and is caused by the fact that if `type(b) ==
    # bytes` then `type(b[0]) == int`, but we need to match each element with a
    # `bytes`, so therefore we make each character an array element
    b = s.encode('utf8')
    bs = [b[i:i+1] for i in range(len(b))]
    if c == bs[0]:
        return parse_specific_string(f, s)
    else:
        return False

def sepBy(p, sep, *args):
    elems = []
    x = optional(p, *args)
    if x != None:
        elems += [x]
        while optional(sep, *args) != None:
            x = p(*args)
            elems += [x]
    return elems

# Assumes '0x' has already been read
def parse_hex_int(f):
    s = b''
    c = get_char(f)
    while c != None:
        if c in string.hexdigits:
            s += c
            c = get_char(f)
        elif c == '_':
            c = get_char(f) # skip _
        else:
            unget_char(f, c)
            break
    return str(int(s, 16))


def parse_int(f):
    s = b''
    c = get_char(f)
    if c == b'0' and peek_char(f) in [b'x', b'X']:
        c = get_char(f) # skip X
        s += parse_hex_int(f)
    else:
        while c != None:
            if c.isdigit():
                s += c
                c = get_char(f)
            elif c == '_':
                c = get_char(f) # skip _
            else:
                unget_char(f, c)
                break
    if len(s) == 0:
        raise ValueError
    return s

def parse_int_signed(f):
    s = b''
    c = get_char(f)

    if c == b'-' and peek_char(f).isdigit():
      s = c + parse_int(f)
    else:
      if c != b'+':
          unget_char(f, c)
      s = parse_int(f)

    return s

def read_str_comma(f):
    skip_spaces(f)
    parse_specific_char(f, b',')
    return b','

def read_str_int(f, s):
    skip_spaces(f)
    x = int(parse_int_signed(f))
    optional_specific_string(f, s)
    return x

def read_str_uint(f, s):
    skip_spaces(f)
    x = int(parse_int(f))
    optional_specific_string(f, s)
    return x

def read_str_i8(f):
    return read_str_int(f, 'i8')
def read_str_i16(f):
    return read_str_int(f, 'i16')
def read_str_i32(f):
    return read_str_int(f, 'i32')
def read_str_i64(f):
    return read_str_int(f, 'i64')

def read_str_u8(f):
    return read_str_int(f, 'u8')
def read_str_u16(f):
    return read_str_int(f, 'u16')
def read_str_u32(f):
    return read_str_int(f, 'u32')
def read_str_u64(f):
    return read_str_int(f, 'u64')

def read_char(f):
    skip_spaces(f)
    parse_specific_char(f, b'\'')
    c = get_char(f)
    parse_specific_char(f, b'\'')
    return c

def read_str_hex_float(f, sign):
    int_part = parse_hex_int(f)
    parse_specific_char(f, b'.')
    frac_part = parse_hex_int(f)
    parse_specific_char(f, b'p')
    exponent = parse_int(f)

    int_val = int(int_part, 16)
    frac_val = float(int(frac_part, 16)) / (16 ** len(frac_part))
    exp_val = int(exponent)

    total_val = (int_val + frac_val) * (2.0 ** exp_val)
    if sign == b'-':
        total_val = -1 * total_val

    return float(total_val)


def read_str_decimal(f):
    skip_spaces(f)
    c = get_char(f)
    if (c == b'-'):
      sign = b'-'
    else:
      unget_char(f,c)
      sign = b''

    # Check for hexadecimal float
    c = get_char(f)
    if (c == '0' and (peek_char(f) in ['x', 'X'])):
        get_char(f)
        return read_str_hex_float(f, sign)
    else:
        unget_char(f, c)

    bef = optional(parse_int, f)
    if bef == None:
        bef = b'0'
        parse_specific_char(f, b'.')
        aft = parse_int(f)
    elif optional(parse_specific_char, f, b'.'):
        aft = parse_int(f)
    else:
        aft = b'0'
    if (optional(parse_specific_char, f, b'E') or
        optional(parse_specific_char, f, b'e')):
        expt = parse_int_signed(f)
    else:
        expt = b'0'
    return float(sign + bef + b'.' + aft + b'E' + expt)

def read_str_f32(f):
    x = read_str_decimal(f)
    optional_specific_string(f, 'f32')
    return x

def read_str_f64(f):
    x = read_str_decimal(f)
    optional_specific_string(f, 'f64')
    return x

def read_str_bool(f):
    skip_spaces(f)
    if peek_char(f) == b't':
        parse_specific_string(f, 'true')
        return True
    elif peek_char(f) == b'f':
        parse_specific_string(f, 'false')
        return False
    else:
        raise ValueError

def read_str_empty_array(f, type_name, rank):
    parse_specific_string(f, 'empty')
    parse_specific_char(f, b'(')
    for i in range(rank):
        parse_specific_string(f, '[]')
    parse_specific_string(f, type_name)
    parse_specific_char(f, b')')

    return None

def read_str_array_elems(f, elem_reader, type_name, rank):
    skip_spaces(f)
    try:
        parse_specific_char(f, b'[')
    except ValueError:
        return read_str_empty_array(f, type_name, rank)
    else:
        xs = sepBy(elem_reader, read_str_comma, f)
        skip_spaces(f)
        parse_specific_char(f, b']')
        return xs

def read_str_array_helper(f, elem_reader, type_name, rank):
    def nested_row_reader(_):
        return read_str_array_helper(f, elem_reader, type_name, rank-1)
    if rank == 1:
        row_reader = elem_reader
    else:
        row_reader = nested_row_reader
    return read_str_array_elems(f, row_reader, type_name, rank-1)

def expected_array_dims(l, rank):
  if rank > 1:
      n = len(l)
      if n == 0:
          elem = []
      else:
          elem = l[0]
      return [n] + expected_array_dims(elem, rank-1)
  else:
      return [len(l)]

def verify_array_dims(l, dims):
    if dims[0] != len(l):
        raise ValueError
    if len(dims) > 1:
        for x in l:
            verify_array_dims(x, dims[1:])

def read_str_array(f, elem_reader, type_name, rank, bt):
    elems = read_str_array_helper(f, elem_reader, type_name, rank)
    if elems == None:
        # Empty array
        return np.empty([0]*rank, dtype=bt)
    else:
        dims = expected_array_dims(elems, rank)
        verify_array_dims(elems, dims)
        return np.array(elems, dtype=bt)

################################################################################

READ_BINARY_VERSION = 2

# struct format specified at
# https://docs.python.org/2/library/struct.html#format-characters

FUTHARK_INT8 = 0
FUTHARK_INT16 = 1
FUTHARK_INT32 = 2
FUTHARK_INT64 = 3
FUTHARK_UINT8 = 4
FUTHARK_UINT16 = 5
FUTHARK_UINT32 = 6
FUTHARK_UINT64 = 7
FUTHARK_FLOAT32 = 8
FUTHARK_FLOAT64 = 9
FUTHARK_BOOL = 10

def mk_bin_scalar_reader(t):
    def bin_reader(f):
        fmt = FUTHARK_PRIMTYPES[t]['bin_format']
        size = FUTHARK_PRIMTYPES[t]['size']
        return struct.unpack('<' + fmt, get_chars(f, size))[0]
    return bin_reader

read_bin_i8 = mk_bin_scalar_reader(FUTHARK_INT8)
read_bin_i16 = mk_bin_scalar_reader(FUTHARK_INT16)
read_bin_i32 = mk_bin_scalar_reader(FUTHARK_INT32)
read_bin_i64 = mk_bin_scalar_reader(FUTHARK_INT64)

read_bin_u8 = mk_bin_scalar_reader(FUTHARK_UINT8)
read_bin_u16 = mk_bin_scalar_reader(FUTHARK_UINT16)
read_bin_u32 = mk_bin_scalar_reader(FUTHARK_UINT32)
read_bin_u64 = mk_bin_scalar_reader(FUTHARK_UINT64)

read_bin_f32 = mk_bin_scalar_reader(FUTHARK_FLOAT32)
read_bin_f64 = mk_bin_scalar_reader(FUTHARK_FLOAT64)

read_bin_bool = mk_bin_scalar_reader(FUTHARK_BOOL)

def read_is_binary(f):
    skip_spaces(f)
    c = get_char(f)
    if c == b'b':
        bin_version = read_bin_u8(f)
        if bin_version != READ_BINARY_VERSION:
            panic(1, "binary-input: File uses version %i, but I only understand version %i.\n",
                  bin_version, READ_BINARY_VERSION)
        return True
    else:
        unget_char(f, c)
        return False

FUTHARK_PRIMTYPES = {}
FUTHARK_PRIMTYPES[FUTHARK_INT8] = \
    {'binname' : b"  i8",
     'type_name' : "i8",
     'size' : 1,
     'bin_reader': read_bin_i8,
     'str_reader': read_str_i8,
     'bin_format': 'b'
    }
FUTHARK_PRIMTYPES[FUTHARK_INT16]   = \
    {'binname' : b" i16",
     'type_name' : "i16",
     'size' : 2,
     'bin_reader': read_bin_i16,
     'str_reader': read_str_i16,
     'bin_format': 'h'
    }
FUTHARK_PRIMTYPES[FUTHARK_INT32]   = \
    {'binname' : b" i32",
     'type_name' : "i32",
     'size' : 4,
     'bin_reader': read_bin_i32,
     'str_reader': read_str_i32,
     'bin_format': 'i'
    }
FUTHARK_PRIMTYPES[FUTHARK_INT64]   = \
    {'binname' : b" i64",
     'type_name' : "i64",
     'size' : 8,
     'bin_reader': read_bin_i64,
     'str_reader': read_str_i64,
     'bin_format': 'q'
    }

FUTHARK_PRIMTYPES[FUTHARK_UINT8] = \
    {'binname' : b"  u8",
     'type_name' : "u8",
     'size' : 1,
     'bin_reader': read_bin_u8,
     'str_reader': read_str_u8,
     'bin_format': 'B'
    }
FUTHARK_PRIMTYPES[FUTHARK_UINT16]   = \
    {'binname' : b" u16",
     'type_name' : "u16",
     'size' : 2,
     'bin_reader': read_bin_u16,
     'str_reader': read_str_u16,
     'bin_format': 'H'
    }
FUTHARK_PRIMTYPES[FUTHARK_UINT32]   = \
    {'binname' : b" u32",
     'type_name' : "u32",
     'size' : 4,
     'bin_reader': read_bin_u32,
     'str_reader': read_str_u32,
     'bin_format': 'I'
    }
FUTHARK_PRIMTYPES[FUTHARK_UINT64]   = \
    {'binname' : b" u64",
     'type_name' : "u64",
     'size' : 8,
     'bin_reader': read_bin_u64,
     'str_reader': read_str_u64,
     'bin_format': 'Q'
    }

FUTHARK_PRIMTYPES[FUTHARK_FLOAT32] = \
    {'binname' : b" f32",
     'type_name' : "f32",
     'size' : 4,
     'bin_reader': read_bin_f32,
     'str_reader': read_str_f32,
     'bin_format': 'f'
    }
FUTHARK_PRIMTYPES[FUTHARK_FLOAT64] = \
    {'binname' : b" f64",
     'type_name' : "f64",
     'size' : 8,
     'bin_reader': read_bin_f64,
     'str_reader': read_str_f64,
     'bin_format': 'd'
    }
FUTHARK_PRIMTYPES[FUTHARK_BOOL]    = \
    {'binname' : b"bool",
     'type_name' : "bool",
     'size' : 1,
     'bin_reader': read_bin_bool,
     'str_reader': read_str_bool,
     'bin_format': 'b'
    }

def read_bin_read_type_enum(f):
    read_binname = get_chars(f, 4)

    for (k,v) in FUTHARK_PRIMTYPES.items():
        if v['binname'] == read_binname:
            return k
    panic(1, "binary-input: Did not recognize the type '%s'.\n", read_binname)

def read_bin_ensure_scalar(f, expected_type):
  dims = read_bin_i8(f)

  if bin_dims != 0:
      panic(1, "binary-input: Expected scalar (0 dimensions), but got array with %i dimensions.\n", bin_dims)

  bin_type_enum = read_bin_read_type_enum(f)
  if bin_type_enum != expected_type:
      panic(1, "binary-input: Expected scalar of type %s but got scalar of type %s.\n",
            FUTHARK_PRIMTYPES[expected_type]['type_name'],
            FUTHARK_PRIMTYPES[bin_type_enum]['type_name'])

# ------------------------------------------------------------------------------
# General interface for reading Primitive Futhark Values
# ------------------------------------------------------------------------------

def read_general(f, ty):
    if read_is_binary(f):
        read_bin_ensure_scalar(ty)
        return FUTHARK_PRIMTYPES[ty]['bin_reader'](f)
    return FUTHARK_PRIMTYPES[ty]['str_reader'](f)

def read_i8(f):
    return read_general(f, FUTHARK_INT8)

def read_i16(f):
    return read_general(f, FUTHARK_INT16)

def read_i32(f):
    return read_general(f, FUTHARK_INT32)

def read_i64(f):
    return read_general(f, FUTHARK_INT64)

def read_u8(f):
    return read_general(f, FUTHARK_UINT8)

def read_u16(f):
    return read_general(f, FUTHARK_UINT16)

def read_u32(f):
    return read_general(f, FUTHARK_UINT32)

def read_u64(f):
    return read_general(f, FUTHARK_UINT64)

def read_f32(f):
    return read_general(f, FUTHARK_FLOAT32)

def read_f64(f):
    return read_general(f, FUTHARK_FLOAT64)

def read_bool(f):
    return read_general(f, FUTHARK_BOOL)

def read_array(f, expected_type, rank, ctype):
    if not read_is_binary(f):
        str_reader = FUTHARK_PRIMTYPES[expected_type]['str_reader']
        return read_str_array(f, str_reader, FUTHARK_PRIMTYPES[expected_type]['type_name'], rank, ctype)

    bin_rank = read_bin_u8(f)

    if bin_rank != rank:
        panic(1, "binary-input: Expected %i dimensions, but got array with %i dimensions.\n",
              rank, bin_rank)

    bin_type_enum = read_bin_read_type_enum(f)
    if expected_type != bin_type_enum:
        panic(1, "binary-input: Expected %iD-array with element type '%s' but got %iD-array with element type '%s'.\n",
              rank, FUTHARK_PRIMTYPES[expected_type]['type_name'],
              rank, FUTHARK_PRIMTYPES[bin_type_enum]['type_name'])

    shape = []
    elem_count = 1
    for i in range(rank):
        bin_size = read_bin_u64(f)
        elem_count *= bin_size
        shape.append(bin_size)

    bin_fmt = FUTHARK_PRIMTYPES[bin_type_enum]['bin_format']

    arr = np.fromfile(f, dtype='<'+bin_fmt, count=elem_count, sep='')
    arr.shape = shape

    return arr

if sys.version_info >= (3,0):
    input_stream = sys.stdin.buffer
else:
    input_stream = sys.stdin

################################################################################
### end of reader.py
################################################################################
# Helper functions dealing with memory blocks.

import ctypes as ct

def addressOffset(x, offset, bt):
  offset = np.asscalar(offset)
  return ct.cast(ct.addressof(x.contents)+offset, ct.POINTER(bt))

def allocateMem(size):
  return ct.cast((ct.c_byte * max(0,size))(), ct.POINTER(ct.c_byte))

# Copy an array if its is not-None.  This is important for treating
# Numpy arrays as flat memory, but has some overhead.
def normaliseArray(x):
  if (x.base is x) or (x.base is None):
    return x
  else:
    return x.copy()

def unwrapArray(x):
  return normaliseArray(x).ctypes.data_as(ct.POINTER(ct.c_byte))

def createArray(x, dim):
  return np.ctypeslib.as_array(x, shape=dim)

def indexArray(x, offset, bt, nptype):
  return nptype(addressOffset(x, offset, bt)[0])

def writeScalarArray(x, offset, v):
  offset = np.asscalar(offset)
  ct.memmove(ct.addressof(x.contents)+offset, ct.addressof(v), ct.sizeof(v))

# An opaque Futhark value.
class opaque(object):
  def __init__(self, desc, *payload):
    self.data = payload
    self.desc = desc

  def __repr__(self):
    return "<opaque Futhark value of type {}>".format(self.desc)
def panic(exitcode, fmt, *args):
    sys.stderr.write('%s: ' % sys.argv[0])
    sys.stderr.write(fmt % args)
    sys.exit(exitcode)
# Scalar functions.

import numpy as np

def signed(x):
  if type(x) == np.uint8:
    return np.int8(x)
  elif type(x) == np.uint16:
    return np.int16(x)
  elif type(x) == np.uint32:
    return np.int32(x)
  else:
    return np.int64(x)

def unsigned(x):
  if type(x) == np.int8:
    return np.uint8(x)
  elif type(x) == np.int16:
    return np.uint16(x)
  elif type(x) == np.int32:
    return np.uint32(x)
  else:
    return np.uint64(x)

def shlN(x,y):
  return x << y

def ashrN(x,y):
  return x >> y

def sdivN(x,y):
  return x // y

def smodN(x,y):
  return x % y

def udivN(x,y):
  return signed(unsigned(x) // unsigned(y))

def umodN(x,y):
  return signed(unsigned(x) % unsigned(y))

def squotN(x,y):
  return np.int32(float(x) / float(y))

def sremN(x,y):
  return np.fmod(x,y)

def sminN(x,y):
  return min(x,y)

def smaxN(x,y):
  return max(x,y)

def uminN(x,y):
  return signed(min(unsigned(x),unsigned(y)))

def umaxN(x,y):
  return signed(max(unsigned(x),unsigned(y)))

def fminN(x,y):
  return min(x,y)

def fmaxN(x,y):
  return max(x,y)

def powN(x,y):
  return x ** y

def fpowN(x,y):
  return x ** y

def sleN(x,y):
  return x <= y

def sltN(x,y):
  return x < y

def uleN(x,y):
  return unsigned(x) <= unsigned(y)

def ultN(x,y):
  return unsigned(x) < unsigned(y)

def lshr8(x,y):
  return np.int8(np.uint8(x) >> np.uint8(y))

def lshr16(x,y):
  return np.int16(np.uint16(x) >> np.uint16(y))

def lshr32(x,y):
  return np.int32(np.uint32(x) >> np.uint32(y))

def lshr64(x,y):
  return np.int64(np.uint64(x) >> np.uint64(y))

def sext_T_i8(x):
  return np.int8(x)

def sext_T_i16(x):
  return np.int16(x)

def sext_T_i32(x):
  return np.int32(x)

def sext_T_i64(x):
  return np.int32(x)

def zext_i8_i8(x):
  return np.int8(np.uint8(x))

def zext_i8_i16(x):
  return np.int16(np.uint8(x))

def zext_i8_i32(x):
  return np.int32(np.uint8(x))

def zext_i8_i64(x):
  return np.int64(np.uint8(x))

def zext_i16_i8(x):
  return np.int8(np.uint16(x))

def zext_i16_i16(x):
  return np.int16(np.uint16(x))

def zext_i16_i32(x):
  return np.int32(np.uint16(x))

def zext_i16_i64(x):
  return np.int64(np.uint16(x))

def zext_i32_i8(x):
  return np.int8(np.uint32(x))

def zext_i32_i16(x):
  return np.int16(np.uint32(x))

def zext_i32_i32(x):
  return np.int32(np.uint32(x))

def zext_i32_i64(x):
  return np.int64(np.uint32(x))

def zext_i64_i8(x):
  return np.int8(np.uint64(x))

def zext_i64_i16(x):
  return np.int16(np.uint64(x))

def zext_i64_i32(x):
  return np.int32(np.uint64(x))

def zext_i64_i64(x):
  return np.int64(np.uint64(x))

shl8 = shl16 = shl32 = shl64 = shlN
ashr8 = ashr16 = ashr32 = ashr64 = ashrN
sdiv8 = sdiv16 = sdiv32 = sdiv64 = sdivN
smod8 = smod16 = smod32 = smod64 = smodN
udiv8 = udiv16 = udiv32 = udiv64 = udivN
umod8 = umod16 = umod32 = umod64 = umodN
squot8 = squot16 = squot32 = squot64 = squotN
srem8 = srem16 = srem32 = srem64 = sremN
smax8 = smax16 = smax32 = smax64 = smaxN
smin8 = smin16 = smin32 = smin64 = sminN
umax8 = umax16 = umax32 = umax64 = umaxN
umin8 = umin16 = umin32 = umin64 = uminN
pow8 = pow16 = pow32 = pow64 = powN
fpow32 = fpow64 = fpowN
fmax32 = fmax64 = fmaxN
fmin32 = fmin64 = fminN
sle8 = sle16 = sle32 = sle64 = sleN
slt8 = slt16 = slt32 = slt64 = sltN
ule8 = ule16 = ule32 = ule64 = uleN
ult8 = ult16 = ult32 = ult64 = ultN
sext_i8_i8 = sext_i16_i8 = sext_i32_i8 = sext_i64_i8 = sext_T_i8
sext_i8_i16 = sext_i16_i16 = sext_i32_i16 = sext_i64_i16 = sext_T_i16
sext_i8_i32 = sext_i16_i32 = sext_i32_i32 = sext_i64_i32 = sext_T_i32
sext_i8_i64 = sext_i16_i64 = sext_i32_i64 = sext_i64_i64 = sext_T_i64

def ssignum(x):
  return np.sign(x)

def usignum(x):
  if x < 0:
    return ssignum(-x)
  else:
    return ssignum(x)

def sitofp_T_f32(x):
  return np.float32(x)
sitofp_i8_f32 = sitofp_i16_f32 = sitofp_i32_f32 = sitofp_i64_f32 = sitofp_T_f32

def sitofp_T_f64(x):
  return np.float64(x)
sitofp_i8_f64 = sitofp_i16_f64 = sitofp_i32_f64 = sitofp_i64_f64 = sitofp_T_f64

def uitofp_T_f32(x):
  return np.float32(unsigned(x))
uitofp_i8_f32 = uitofp_i16_f32 = uitofp_i32_f32 = uitofp_i64_f32 = uitofp_T_f32

def uitofp_T_f64(x):
  return np.float64(unsigned(x))
uitofp_i8_f64 = uitofp_i16_f64 = uitofp_i32_f64 = uitofp_i64_f64 = uitofp_T_f64

def fptosi_T_i8(x):
  return np.int8(np.trunc(x))
fptosi_f32_i8 = fptosi_f64_i8 = fptosi_T_i8

def fptosi_T_i16(x):
  return np.int16(np.trunc(x))
fptosi_f32_i16 = fptosi_f64_i16 = fptosi_T_i16

def fptosi_T_i32(x):
  return np.int32(np.trunc(x))
fptosi_f32_i32 = fptosi_f64_i32 = fptosi_T_i32

def fptosi_T_i64(x):
  return np.int64(np.trunc(x))
fptosi_f32_i64 = fptosi_f64_i64 = fptosi_T_i64

def fptoui_T_i8(x):
  return np.uint8(np.trunc(x))
fptoui_f32_i8 = fptoui_f64_i8 = fptoui_T_i8

def fptoui_T_i16(x):
  return np.uint16(np.trunc(x))
fptoui_f32_i16 = fptoui_f64_i16 = fptoui_T_i16

def fptoui_T_i32(x):
  return np.uint32(np.trunc(x))
fptoui_f32_i32 = fptoui_f64_i32 = fptoui_T_i32

def fptoui_T_i64(x):
  return np.uint64(np.trunc(x))
fptoui_f32_i64 = fptoui_f64_i64 = fptoui_T_i64

def fpconv_f32_f64(x):
  return np.float64(x)

def fpconv_f64_f32(x):
  return np.float32(x)

def futhark_log64(x):
  return np.float64(np.log(x))

def futhark_sqrt64(x):
  return np.sqrt(x)

def futhark_exp64(x):
  return np.exp(x)

def futhark_cos64(x):
  return np.cos(x)

def futhark_sin64(x):
  return np.sin(x)

def futhark_acos64(x):
  return np.arccos(x)

def futhark_asin64(x):
  return np.arcsin(x)

def futhark_atan64(x):
  return np.arctan(x)

def futhark_atan2_64(x, y):
  return np.arctan2(x, y)

def futhark_isnan64(x):
  return np.isnan(x)

def futhark_isinf64(x):
  return np.isinf(x)

def futhark_log32(x):
  return np.float32(np.log(x))

def futhark_sqrt32(x):
  return np.float32(np.sqrt(x))

def futhark_exp32(x):
  return np.exp(x)

def futhark_cos32(x):
  return np.cos(x)

def futhark_sin32(x):
  return np.sin(x)

def futhark_acos32(x):
  return np.arccos(x)

def futhark_asin32(x):
  return np.arcsin(x)

def futhark_atan32(x):
  return np.arctan(x)

def futhark_atan2_32(x, y):
  return np.arctan2(x, y)

def futhark_isnan32(x):
  return np.isnan(x)

def futhark_isinf32(x):
  return np.isinf(x)
class core_shell_parallelepiped_futhark:
  def __init__(self, interactive=False, platform_pref=preferred_platform,
               device_pref=preferred_device, group_size=256, num_groups=128,
               tile_size=32):
    self.ctx = get_prefered_context(interactive, platform_pref, device_pref)
    self.queue = cl.CommandQueue(self.ctx)
    self.device = self.ctx.get_info(cl.context_info.DEVICES)[0]
     # XXX: Assuming just a single device here.
    platform_name = self.ctx.get_info(cl.context_info.DEVICES)[0].platform.name
    device_type = self.device.type
    lockstep_width = 1
    if ((platform_name == "NVIDIA CUDA") and (device_type == cl.device_type.GPU)):
      lockstep_width = np.int32(32)
    if ((platform_name == "AMD Accelerated Parallel Processing") and (device_type == cl.device_type.GPU)):
      lockstep_width = np.int32(64)
    max_tile_size = int(np.sqrt(self.device.max_work_group_size))
    if (tile_size * tile_size > self.device.max_work_group_size):
      sys.stderr.write('Warning: Device limits tile size to {} (setting was {})\n'.format(max_tile_size, tile_size))
      tile_size = max_tile_size
    self.group_size = group_size
    self.num_groups = num_groups
    self.tile_size = tile_size
    if (len(fut_opencl_src) >= 0):
      program = cl.Program(self.ctx, fut_opencl_src).build(["-DFUT_BLOCK_DIM={}".format(FUT_BLOCK_DIM),
                                                            "-DLOCKSTEP_WIDTH={}".format(lockstep_width),
                                                            "-DDEFAULT_GROUP_SIZE={}".format(group_size),
                                                            "-DDEFAULT_NUM_GROUPS={}".format(num_groups),
                                                            "-DDEFAULT_TILE_SIZE={}".format(tile_size)])
    
    self.map_kernel_8918_var = program.map_kernel_8918
    self.map_kernel_9009_var = program.map_kernel_9009
    self.map_kernel_9124_var = program.map_kernel_9124
    self.map_kernel_9247_var = program.map_kernel_9247
    static_array_9397 = np.array([np.float64(-0.999505948362153),
                                  np.float64(-0.997397786355355),
                                  np.float64(-0.993608772723527),
                                  np.float64(-0.988144453359837),
                                  np.float64(-0.981013938975656),
                                  np.float64(-0.972229228520377),
                                  np.float64(-0.961805126758768),
                                  np.float64(-0.949759207710896),
                                  np.float64(-0.936111781934811),
                                  np.float64(-0.92088586125215),
                                  np.float64(-0.904107119545567),
                                  np.float64(-0.885803849292083),
                                  np.float64(-0.866006913771982),
                                  np.float64(-0.844749694983342),
                                  np.float64(-0.822068037328975),
                                  np.float64(-0.7980001871612),
                                  np.float64(-0.77258672828181),
                                  np.float64(-0.74587051350361),
                                  np.float64(-0.717896592387704),
                                  np.float64(-0.688712135277641),
                                  np.float64(-0.658366353758143),
                                  np.float64(-0.626910417672267),
                                  np.float64(-0.594397368836793),
                                  np.float64(-0.560882031601237),
                                  np.float64(-0.526420920401243),
                                  np.float64(-0.491072144462194),
                                  np.float64(-0.454895309813726),
                                  np.float64(-0.417951418780327),
                                  np.float64(-0.380302767117504),
                                  np.float64(-0.342012838966962),
                                  np.float64(-0.303146199807908),
                                  np.float64(-0.263768387584994),
                                  np.float64(-0.223945802196474),
                                  np.float64(-0.183745593528914),
                                  np.float64(-0.143235548227268),
                                  np.float64(-0.102483975391227),
                                  np.float64(-6.15595913906112e-2),
                                  np.float64(-2.05314039939986e-2),
                                  np.float64(2.05314039939986e-2),
                                  np.float64(6.15595913906112e-2),
                                  np.float64(0.102483975391227),
                                  np.float64(0.143235548227268),
                                  np.float64(0.183745593528914),
                                  np.float64(0.223945802196474),
                                  np.float64(0.263768387584994),
                                  np.float64(0.303146199807908),
                                  np.float64(0.342012838966962),
                                  np.float64(0.380302767117504),
                                  np.float64(0.417951418780327),
                                  np.float64(0.454895309813726),
                                  np.float64(0.491072144462194),
                                  np.float64(0.526420920401243),
                                  np.float64(0.560882031601237),
                                  np.float64(0.594397368836793),
                                  np.float64(0.626910417672267),
                                  np.float64(0.658366353758143),
                                  np.float64(0.688712135277641),
                                  np.float64(0.717896592387704),
                                  np.float64(0.74587051350361),
                                  np.float64(0.77258672828181),
                                  np.float64(0.7980001871612),
                                  np.float64(0.822068037328975),
                                  np.float64(0.844749694983342),
                                  np.float64(0.866006913771982),
                                  np.float64(0.885803849292083),
                                  np.float64(0.904107119545567),
                                  np.float64(0.92088586125215),
                                  np.float64(0.936111781934811),
                                  np.float64(0.949759207710896),
                                  np.float64(0.961805126758768),
                                  np.float64(0.972229228520377),
                                  np.float64(0.981013938975656),
                                  np.float64(0.988144453359837),
                                  np.float64(0.993608772723527),
                                  np.float64(0.997397786355355),
                                  np.float64(0.999505948362153)],
                                 dtype=np.float64)
    static_mem_9434 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                np.long(np.long(np.int32(608)) if (np.int32(608) > np.int32(0)) else np.int32(1)))
    if (np.int32(608) != np.int32(0)):
      cl.enqueue_copy(self.queue, static_mem_9434,
                      normaliseArray(static_array_9397),
                      is_blocking=synchronous)
    self.static_array_9397 = static_mem_9434
    static_array_9398 = np.array([np.float64(1.26779163408536e-3),
                                  np.float64(2.94910295364247e-3),
                                  np.float64(4.62793522803742e-3),
                                  np.float64(6.29918049732845e-3),
                                  np.float64(7.95984747723973e-3),
                                  np.float64(9.60710541471375e-3),
                                  np.float64(1.12381685696677e-2),
                                  np.float64(1.28502838475101e-2),
                                  np.float64(1.44407317482767e-2),
                                  np.float64(1.60068299122486e-2),
                                  np.float64(1.75459372914742e-2),
                                  np.float64(1.90554584671906e-2),
                                  np.float64(2.0532847967908e-2),
                                  np.float64(2.19756145344162e-2),
                                  np.float64(2.33813253070112e-2),
                                  np.float64(2.47476099206597e-2),
                                  np.float64(2.6072164497986e-2),
                                  np.float64(2.73527555318275e-2),
                                  np.float64(2.8587223650054e-2),
                                  np.float64(2.9773487255905e-2),
                                  np.float64(3.09095460374916e-2),
                                  np.float64(3.19934843404216e-2),
                                  np.float64(3.30234743977917e-2),
                                  np.float64(3.39977794120564e-2),
                                  np.float64(3.49147564835508e-2),
                                  np.float64(3.57728593807139e-2),
                                  np.float64(3.65706411473296e-2),
                                  np.float64(3.73067565423816e-2),
                                  np.float64(3.79799643084053e-2),
                                  np.float64(3.85891292645067e-2),
                                  np.float64(3.91332242205184e-2),
                                  np.float64(3.96113317090621e-2),
                                  np.float64(4.00226455325968e-2),
                                  np.float64(4.0366472122844e-2),
                                  np.float64(4.06422317102947e-2),
                                  np.float64(4.08494593018285e-2),
                                  np.float64(4.0987805464794e-2),
                                  np.float64(4.10570369162294e-2),
                                  np.float64(4.10570369162294e-2),
                                  np.float64(4.0987805464794e-2),
                                  np.float64(4.08494593018285e-2),
                                  np.float64(4.06422317102947e-2),
                                  np.float64(4.0366472122844e-2),
                                  np.float64(4.00226455325968e-2),
                                  np.float64(3.96113317090621e-2),
                                  np.float64(3.91332242205184e-2),
                                  np.float64(3.85891292645067e-2),
                                  np.float64(3.79799643084053e-2),
                                  np.float64(3.73067565423816e-2),
                                  np.float64(3.65706411473296e-2),
                                  np.float64(3.57728593807139e-2),
                                  np.float64(3.49147564835508e-2),
                                  np.float64(3.39977794120564e-2),
                                  np.float64(3.30234743977917e-2),
                                  np.float64(3.19934843404216e-2),
                                  np.float64(3.09095460374916e-2),
                                  np.float64(2.9773487255905e-2),
                                  np.float64(2.8587223650054e-2),
                                  np.float64(2.73527555318275e-2),
                                  np.float64(2.6072164497986e-2),
                                  np.float64(2.47476099206597e-2),
                                  np.float64(2.33813253070112e-2),
                                  np.float64(2.19756145344162e-2),
                                  np.float64(2.0532847967908e-2),
                                  np.float64(1.90554584671906e-2),
                                  np.float64(1.75459372914742e-2),
                                  np.float64(1.60068299122486e-2),
                                  np.float64(1.44407317482767e-2),
                                  np.float64(1.28502838475101e-2),
                                  np.float64(1.12381685696677e-2),
                                  np.float64(9.60710541471375e-3),
                                  np.float64(7.95984747723973e-3),
                                  np.float64(6.29918049732845e-3),
                                  np.float64(4.62793522803742e-3),
                                  np.float64(2.94910295364247e-3),
                                  np.float64(1.26779163408536e-3)],
                                 dtype=np.float64)
    static_mem_9435 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                np.long(np.long(np.int32(608)) if (np.int32(608) > np.int32(0)) else np.int32(1)))
    if (np.int32(608) != np.int32(0)):
      cl.enqueue_copy(self.queue, static_mem_9435,
                      normaliseArray(static_array_9398),
                      is_blocking=synchronous)
    self.static_array_9398 = static_mem_9435
  def futhark_kernel_float64_2d(self, call_details_buffer_mem_sizze_9351,
                                values_mem_sizze_9353, qx_input_mem_sizze_9355,
                                qy_input_mem_sizze_9357,
                                call_details_buffer_mem_9352, values_mem_9354,
                                qx_input_mem_9356, qy_input_mem_9358,
                                sizze_7630, sizze_7631, sizze_7632, sizze_7633,
                                num_pars_7634, num_active_7635, nq_7636,
                                call_details_num_evals_7637, cutoff_7642):
    y_7643 = slt32(np.int32(1), sizze_7631)
    assert y_7643, 'At ./header/for_iq.fut:58:22-58:22: index out of bounds'
    read_res_9405 = np.empty(np.int32(1), dtype=ct.c_double)
    cl.enqueue_copy(self.queue, read_res_9405, values_mem_9354,
                    device_offset=np.long(np.int64(8)), is_blocking=True)
    res_7645 = read_res_9405[np.int32(0)]
    j_7646 = (num_pars_7634 + np.int32(2))
    x_7647 = abs(num_pars_7634)
    empty_slice_7648 = (x_7647 == np.int32(0))
    m_7649 = (x_7647 - np.int32(1))
    i_p_m_t_s_7650 = (np.int32(2) + m_7649)
    zzero_leq_i_p_m_t_s_7651 = sle32(np.int32(0), i_p_m_t_s_7650)
    i_p_m_t_s_leq_w_7652 = slt32(i_p_m_t_s_7650, sizze_7631)
    i_lte_j_7653 = sle32(np.int32(2), j_7646)
    y_7654 = (zzero_leq_i_p_m_t_s_7651 and i_p_m_t_s_leq_w_7652)
    y_7655 = (i_lte_j_7653 and y_7654)
    ok_or_empty_7656 = (empty_slice_7648 or y_7655)
    assert ok_or_empty_7656, 'At ./header/for_iq.fut:59:22-59:22: slice out of bounds'
    y_7661 = slt32(np.int32(0), x_7647)
    assert y_7661, 'At core_shell_parallelepiped_futhark.fut:78:20-78:20: index out of bounds'
    read_res_9406 = np.empty(np.int32(1), dtype=ct.c_double)
    cl.enqueue_copy(self.queue, read_res_9406, values_mem_9354,
                    device_offset=np.long(np.int32(16)), is_blocking=True)
    res_7663 = read_res_9406[np.int32(0)]
    y_7664 = slt32(np.int32(1), x_7647)
    assert y_7664, 'At core_shell_parallelepiped_futhark.fut:79:20-79:20: index out of bounds'
    read_res_9407 = np.empty(np.int32(1), dtype=ct.c_double)
    cl.enqueue_copy(self.queue, read_res_9407, values_mem_9354,
                    device_offset=np.long(np.int32(24)), is_blocking=True)
    res_7666 = read_res_9407[np.int32(0)]
    y_7667 = slt32(np.int32(2), x_7647)
    assert y_7667, 'At core_shell_parallelepiped_futhark.fut:80:20-80:20: index out of bounds'
    read_res_9408 = np.empty(np.int32(1), dtype=ct.c_double)
    cl.enqueue_copy(self.queue, read_res_9408, values_mem_9354,
                    device_offset=np.long(np.int32(32)), is_blocking=True)
    res_7669 = read_res_9408[np.int32(0)]
    y_7670 = slt32(np.int32(3), x_7647)
    assert y_7670, 'At core_shell_parallelepiped_futhark.fut:81:20-81:20: index out of bounds'
    read_res_9409 = np.empty(np.int32(1), dtype=ct.c_double)
    cl.enqueue_copy(self.queue, read_res_9409, values_mem_9354,
                    device_offset=np.long(np.int32(40)), is_blocking=True)
    res_7672 = read_res_9409[np.int32(0)]
    y_7673 = slt32(np.int32(4), x_7647)
    assert y_7673, 'At core_shell_parallelepiped_futhark.fut:82:23-82:23: index out of bounds'
    read_res_9410 = np.empty(np.int32(1), dtype=ct.c_double)
    cl.enqueue_copy(self.queue, read_res_9410, values_mem_9354,
                    device_offset=np.long(np.int32(48)), is_blocking=True)
    res_7675 = read_res_9410[np.int32(0)]
    y_7676 = slt32(np.int32(5), x_7647)
    assert y_7676, 'At core_shell_parallelepiped_futhark.fut:83:20-83:20: index out of bounds'
    read_res_9411 = np.empty(np.int32(1), dtype=ct.c_double)
    cl.enqueue_copy(self.queue, read_res_9411, values_mem_9354,
                    device_offset=np.long(np.int32(56)), is_blocking=True)
    res_7678 = read_res_9411[np.int32(0)]
    y_7679 = slt32(np.int32(6), x_7647)
    assert y_7679, 'At core_shell_parallelepiped_futhark.fut:84:20-84:20: index out of bounds'
    read_res_9412 = np.empty(np.int32(1), dtype=ct.c_double)
    cl.enqueue_copy(self.queue, read_res_9412, values_mem_9354,
                    device_offset=np.long(np.int32(64)), is_blocking=True)
    res_7681 = read_res_9412[np.int32(0)]
    y_7682 = slt32(np.int32(7), x_7647)
    assert y_7682, 'At core_shell_parallelepiped_futhark.fut:85:20-85:20: index out of bounds'
    read_res_9413 = np.empty(np.int32(1), dtype=ct.c_double)
    cl.enqueue_copy(self.queue, read_res_9413, values_mem_9354,
                    device_offset=np.long(np.int32(72)), is_blocking=True)
    res_7684 = read_res_9413[np.int32(0)]
    y_7685 = slt32(np.int32(8), x_7647)
    assert y_7685, 'At core_shell_parallelepiped_futhark.fut:86:23-86:23: index out of bounds'
    read_res_9414 = np.empty(np.int32(1), dtype=ct.c_double)
    cl.enqueue_copy(self.queue, read_res_9414, values_mem_9354,
                    device_offset=np.long(np.int32(80)), is_blocking=True)
    res_7687 = read_res_9414[np.int32(0)]
    y_7688 = slt32(np.int32(9), x_7647)
    assert y_7688, 'At core_shell_parallelepiped_futhark.fut:87:23-87:23: index out of bounds'
    read_res_9415 = np.empty(np.int32(1), dtype=ct.c_double)
    cl.enqueue_copy(self.queue, read_res_9415, values_mem_9354,
                    device_offset=np.long(np.int32(88)), is_blocking=True)
    res_7690 = read_res_9415[np.int32(0)]
    y_7691 = slt32(np.int32(10), x_7647)
    assert y_7691, 'At core_shell_parallelepiped_futhark.fut:88:23-88:23: index out of bounds'
    read_res_9416 = np.empty(np.int32(1), dtype=ct.c_double)
    cl.enqueue_copy(self.queue, read_res_9416, values_mem_9354,
                    device_offset=np.long(np.int32(96)), is_blocking=True)
    res_7693 = read_res_9416[np.int32(0)]
    y_7694 = slt32(np.int32(11), x_7647)
    assert y_7694, 'At core_shell_parallelepiped_futhark.fut:89:17-89:17: index out of bounds'
    read_res_9417 = np.empty(np.int32(1), dtype=ct.c_double)
    cl.enqueue_copy(self.queue, read_res_9417, values_mem_9354,
                    device_offset=np.long(np.int32(104)), is_blocking=True)
    res_7696 = read_res_9417[np.int32(0)]
    y_7697 = slt32(np.int32(12), x_7647)
    assert y_7697, 'At core_shell_parallelepiped_futhark.fut:90:15-90:15: index out of bounds'
    read_res_9418 = np.empty(np.int32(1), dtype=ct.c_double)
    cl.enqueue_copy(self.queue, read_res_9418, values_mem_9354,
                    device_offset=np.long(np.int32(112)), is_blocking=True)
    res_7699 = read_res_9418[np.int32(0)]
    y_7700 = slt32(np.int32(13), x_7647)
    assert y_7700, 'At core_shell_parallelepiped_futhark.fut:91:15-91:15: index out of bounds'
    read_res_9419 = np.empty(np.int32(1), dtype=ct.c_double)
    cl.enqueue_copy(self.queue, read_res_9419, values_mem_9354,
                    device_offset=np.long(np.int32(120)), is_blocking=True)
    res_7702 = read_res_9419[np.int32(0)]
    res_7703 = (res_7663 - res_7675)
    res_7704 = (res_7666 - res_7675)
    res_7705 = (res_7669 - res_7675)
    res_7706 = (res_7672 - res_7675)
    res_7707 = (res_7696 * np.float64(1.7453292519943295e-2))
    res_7708 = futhark_sin64(res_7707)
    res_7709 = futhark_cos64(res_7707)
    res_7710 = (res_7699 * np.float64(1.7453292519943295e-2))
    res_7711 = futhark_sin64(res_7710)
    res_7712 = futhark_cos64(res_7710)
    res_7713 = (res_7702 * np.float64(1.7453292519943295e-2))
    res_7714 = futhark_sin64(res_7713)
    res_7715 = futhark_cos64(res_7713)
    res_7716 = (np.float64(0.0) - res_7711)
    res_7717 = (res_7716 * res_7714)
    res_7718 = (res_7709 * res_7712)
    res_7719 = (res_7718 * res_7715)
    res_7720 = (res_7717 + res_7719)
    res_7721 = (res_7712 * res_7714)
    res_7722 = (res_7709 * res_7711)
    res_7723 = (res_7722 * res_7715)
    res_7724 = (res_7721 + res_7723)
    res_7725 = (res_7716 * res_7715)
    res_7726 = (res_7718 * res_7714)
    res_7727 = (res_7725 - res_7726)
    res_7728 = (res_7712 * res_7715)
    res_7729 = (res_7722 * res_7714)
    res_7730 = (res_7728 - res_7729)
    res_7731 = (np.float64(0.0) - res_7708)
    res_7732 = (res_7731 * res_7712)
    res_7733 = (res_7731 * res_7711)
    res_7734 = (res_7678 * res_7681)
    res_7735 = (res_7734 * res_7684)
    res_7736 = (np.float64(2.0) * res_7687)
    res_7737 = (res_7736 * res_7681)
    res_7738 = (res_7737 * res_7684)
    res_7739 = (np.float64(2.0) * res_7678)
    res_7740 = (res_7739 * res_7690)
    res_7741 = (res_7740 * res_7684)
    res_7742 = (res_7739 * res_7681)
    res_7743 = (res_7742 * res_7693)
    res_7744 = (res_7678 + res_7736)
    res_7745 = (np.float64(2.0) * res_7690)
    res_7746 = (res_7678 + res_7745)
    res_7747 = (np.float64(2.0) * res_7693)
    res_7748 = (res_7678 + res_7747)
    cond_7829 = (num_active_7635 == np.int32(0))
    x_7830 = abs(nq_7636)
    if cond_7829:
      branch_ctx_7831 = x_7830
    else:
      branch_ctx_7831 = nq_7636
    bounds_invalid_upwards_7832 = slt32(nq_7636, np.int32(0))
    binop_x_9360 = sext_i32_i64(sizze_7632)
    bytes_9359 = (binop_x_9360 * np.int64(8))
    mem_9361 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         np.long(np.long(bytes_9359) if (bytes_9359 > np.int32(0)) else np.int32(1)))
    binop_x_9363 = sext_i32_i64(nq_7636)
    bytes_9362 = (binop_x_9363 * np.int64(8))
    mem_9364 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         np.long(np.long(bytes_9362) if (bytes_9362 > np.int32(0)) else np.int32(1)))
    if cond_7829:
      res_7857 = (res_7735 + res_7738)
      res_7859 = (res_7745 * res_7678)
      res_7860 = (res_7859 * res_7684)
      res_7861 = (res_7857 + res_7860)
      res_7863 = (res_7747 * res_7678)
      res_7864 = (res_7863 * res_7681)
      res_7865 = (res_7861 + res_7864)
      res_7866 = (res_7865 == np.float64(0.0))
      if res_7866:
        y_7868 = slt32(np.int32(0), sizze_7631)
        assert y_7868, 'At ./header/for_iq.fut:64:55-64:55: index out of bounds'
        read_res_9420 = np.empty(np.int32(1), dtype=ct.c_double)
        cl.enqueue_copy(self.queue, read_res_9420, values_mem_9354,
                        device_offset=np.long(np.int32(0)), is_blocking=True)
        res_7870 = read_res_9420[np.int32(0)]
        res_7867 = res_7870
      else:
        y_7871 = slt32(np.int32(0), sizze_7631)
        assert y_7871, 'At ./header/for_iq.fut:64:70-64:70: index out of bounds'
        read_res_9421 = np.empty(np.int32(1), dtype=ct.c_double)
        cl.enqueue_copy(self.queue, read_res_9421, values_mem_9354,
                        device_offset=np.long(np.int32(0)), is_blocking=True)
        arg_7873 = read_res_9421[np.int32(0)]
        res_7874 = (arg_7873 / res_7865)
        res_7867 = res_7874
      group_sizze_8913 = self.group_size
      y_8914 = (group_sizze_8913 - np.int32(1))
      x_8915 = (sizze_7632 + y_8914)
      num_groups_8916 = squot32(x_8915, group_sizze_8913)
      num_threads_8917 = (num_groups_8916 * group_sizze_8913)
      if ((np.int32(1) * (num_groups_8916 * group_sizze_8913)) != np.int32(0)):
        self.map_kernel_8918_var.set_args(np.int32(sizze_7632),
                                          np.float64(res_7645),
                                          np.float64(res_7678),
                                          np.float64(res_7681),
                                          np.float64(res_7684),
                                          np.float64(res_7703),
                                          np.float64(res_7704),
                                          np.float64(res_7705),
                                          np.float64(res_7706),
                                          np.float64(res_7720),
                                          np.float64(res_7724),
                                          np.float64(res_7727),
                                          np.float64(res_7730),
                                          np.float64(res_7732),
                                          np.float64(res_7733),
                                          np.float64(res_7735),
                                          np.float64(res_7738),
                                          np.float64(res_7741),
                                          np.float64(res_7743),
                                          np.float64(res_7744),
                                          np.float64(res_7746),
                                          np.float64(res_7748),
                                          np.float64(res_7867),
                                          qx_input_mem_9356, qy_input_mem_9358,
                                          mem_9361)
        cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_8918_var,
                                   (np.long((num_groups_8916 * group_sizze_8913)),),
                                   (np.long(group_sizze_8913),))
        if synchronous:
          self.queue.finish()
      empty_slice_7959 = (x_7830 == np.int32(0))
      m_7960 = (x_7830 - np.int32(1))
      zzero_leq_i_p_m_t_s_7961 = sle32(np.int32(0), m_7960)
      i_p_m_t_s_leq_w_7962 = slt32(m_7960, sizze_7632)
      i_lte_j_7963 = sle32(np.int32(0), nq_7636)
      y_7964 = (zzero_leq_i_p_m_t_s_7961 and i_p_m_t_s_leq_w_7962)
      y_7965 = (i_lte_j_7963 and y_7964)
      ok_or_empty_7966 = (empty_slice_7959 or y_7965)
      assert ok_or_empty_7966, 'At ./header/for_iq.fut:69:8-69:8: slice out of bounds'
      res_mem_9366 = mem_9361
      res_mem_sizze_9365 = bytes_9359
    else:
      not_p_7970 = not(bounds_invalid_upwards_7832)
      assert not_p_7970, 'At ./futlib/array.fut:42:5-42:5: shape of function result does not match shapes in return type'
      res_7972 = (nq_7636 + np.int32(1))
      res_7973 = sdiv32(np.int32(1000000), res_7972)
      res_7974 = smax32(res_7973, np.int32(1))
      res_7976 = (cutoff_7642 < np.float64(1.0))
      r_7977 = np.float64(0.0)
      x_7982 = (np.int32(2) * res_7974)
      cond_7983 = slt32(x_7982, call_details_num_evals_7637)
      if cond_7983:
        y_7985 = slt32(np.int32(0), sizze_7631)
        assert y_7985, 'At ./header/for_iq.fut:88:57-88:57: index out of bounds'
        read_res_9422 = np.empty(np.int32(1), dtype=ct.c_double)
        cl.enqueue_copy(self.queue, read_res_9422, values_mem_9354,
                        device_offset=np.long(np.int32(0)), is_blocking=True)
        arg_7987 = read_res_9422[np.int32(0)]
        arg_7988 = (call_details_num_evals_7637 - res_7974)
        res_7989 = sitofp_i32_f64(arg_7988)
        res_7990 = (arg_7987 / res_7989)
        res_7984 = res_7990
      else:
        y_7991 = slt32(np.int32(0), sizze_7631)
        assert y_7991, 'At ./header/for_iq.fut:88:125-88:125: index out of bounds'
        read_res_9423 = np.empty(np.int32(1), dtype=ct.c_double)
        cl.enqueue_copy(self.queue, read_res_9423, values_mem_9354,
                        device_offset=np.long(np.int32(0)), is_blocking=True)
        res_7993 = read_res_9423[np.int32(0)]
        res_7984 = res_7993
      group_sizze_9004 = self.group_size
      y_9005 = (group_sizze_9004 - np.int32(1))
      x_9006 = (nq_7636 + y_9005)
      num_groups_9007 = squot32(x_9006, group_sizze_9004)
      num_threads_9008 = (num_groups_9007 * group_sizze_9004)
      if ((np.int32(1) * (num_groups_9007 * group_sizze_9004)) != np.int32(0)):
        self.map_kernel_9009_var.set_args(np.int32(nq_7636),
                                          np.float64(res_7645),
                                          np.float64(res_7678),
                                          np.float64(res_7681),
                                          np.float64(res_7684),
                                          np.float64(res_7703),
                                          np.float64(res_7704),
                                          np.float64(res_7705),
                                          np.float64(res_7706),
                                          np.float64(res_7720),
                                          np.float64(res_7724),
                                          np.float64(res_7727),
                                          np.float64(res_7730),
                                          np.float64(res_7732),
                                          np.float64(res_7733),
                                          np.float64(res_7735),
                                          np.float64(res_7738),
                                          np.float64(res_7741),
                                          np.float64(res_7743),
                                          np.float64(res_7744),
                                          np.float64(res_7746),
                                          np.float64(res_7748),
                                          np.byte(res_7976),
                                          np.float64(res_7984),
                                          qx_input_mem_9356, qy_input_mem_9358,
                                          mem_9364)
        cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_9009_var,
                                   (np.long((num_groups_9007 * group_sizze_9004)),),
                                   (np.long(group_sizze_9004),))
        if synchronous:
          self.queue.finish()
      res_mem_9366 = mem_9364
      res_mem_sizze_9365 = bytes_9362
    out_mem_9385 = res_mem_9366
    out_arrsizze_9387 = branch_ctx_7831
    out_memsizze_9386 = res_mem_sizze_9365
    return (out_memsizze_9386, out_mem_9385, out_arrsizze_9387)
  def futhark_kernel_float64(self, call_details_buffer_mem_sizze_9351,
                             values_mem_sizze_9353, q_input_mem_sizze_9355,
                             call_details_buffer_mem_9352, values_mem_9354,
                             q_input_mem_9356, sizze_8083, sizze_8084,
                             sizze_8085, num_pars_8086, num_active_8087,
                             nq_8088, call_details_num_evals_8089, cutoff_8093):
    y_8094 = slt32(np.int32(1), sizze_8084)
    assert y_8094, 'At ./header/for_iq.fut:16:22-16:22: index out of bounds'
    read_res_9424 = np.empty(np.int32(1), dtype=ct.c_double)
    cl.enqueue_copy(self.queue, read_res_9424, values_mem_9354,
                    device_offset=np.long(np.int64(8)), is_blocking=True)
    res_8096 = read_res_9424[np.int32(0)]
    j_8097 = (num_pars_8086 + np.int32(2))
    x_8098 = abs(num_pars_8086)
    empty_slice_8099 = (x_8098 == np.int32(0))
    m_8100 = (x_8098 - np.int32(1))
    i_p_m_t_s_8101 = (np.int32(2) + m_8100)
    zzero_leq_i_p_m_t_s_8102 = sle32(np.int32(0), i_p_m_t_s_8101)
    i_p_m_t_s_leq_w_8103 = slt32(i_p_m_t_s_8101, sizze_8084)
    i_lte_j_8104 = sle32(np.int32(2), j_8097)
    y_8105 = (zzero_leq_i_p_m_t_s_8102 and i_p_m_t_s_leq_w_8103)
    y_8106 = (i_lte_j_8104 and y_8105)
    ok_or_empty_8107 = (empty_slice_8099 or y_8106)
    assert ok_or_empty_8107, 'At ./header/for_iq.fut:17:22-17:22: slice out of bounds'
    y_8109 = slt32(np.int32(0), x_8098)
    assert y_8109, 'At core_shell_parallelepiped_futhark.fut:22:20-22:20: index out of bounds'
    read_res_9425 = np.empty(np.int32(1), dtype=ct.c_double)
    cl.enqueue_copy(self.queue, read_res_9425, values_mem_9354,
                    device_offset=np.long(np.int32(16)), is_blocking=True)
    res_8111 = read_res_9425[np.int32(0)]
    y_8112 = slt32(np.int32(1), x_8098)
    assert y_8112, 'At core_shell_parallelepiped_futhark.fut:23:20-23:20: index out of bounds'
    read_res_9426 = np.empty(np.int32(1), dtype=ct.c_double)
    cl.enqueue_copy(self.queue, read_res_9426, values_mem_9354,
                    device_offset=np.long(np.int32(24)), is_blocking=True)
    res_8114 = read_res_9426[np.int32(0)]
    y_8115 = slt32(np.int32(2), x_8098)
    assert y_8115, 'At core_shell_parallelepiped_futhark.fut:24:20-24:20: index out of bounds'
    read_res_9427 = np.empty(np.int32(1), dtype=ct.c_double)
    cl.enqueue_copy(self.queue, read_res_9427, values_mem_9354,
                    device_offset=np.long(np.int32(32)), is_blocking=True)
    res_8117 = read_res_9427[np.int32(0)]
    y_8118 = slt32(np.int32(4), x_8098)
    assert y_8118, 'At core_shell_parallelepiped_futhark.fut:26:23-26:23: index out of bounds'
    read_res_9428 = np.empty(np.int32(1), dtype=ct.c_double)
    cl.enqueue_copy(self.queue, read_res_9428, values_mem_9354,
                    device_offset=np.long(np.int32(48)), is_blocking=True)
    res_8120 = read_res_9428[np.int32(0)]
    y_8121 = slt32(np.int32(5), x_8098)
    assert y_8121, 'At core_shell_parallelepiped_futhark.fut:27:20-27:20: index out of bounds'
    read_res_9429 = np.empty(np.int32(1), dtype=ct.c_double)
    cl.enqueue_copy(self.queue, read_res_9429, values_mem_9354,
                    device_offset=np.long(np.int32(56)), is_blocking=True)
    res_8123 = read_res_9429[np.int32(0)]
    y_8124 = slt32(np.int32(6), x_8098)
    assert y_8124, 'At core_shell_parallelepiped_futhark.fut:28:20-28:20: index out of bounds'
    read_res_9430 = np.empty(np.int32(1), dtype=ct.c_double)
    cl.enqueue_copy(self.queue, read_res_9430, values_mem_9354,
                    device_offset=np.long(np.int32(64)), is_blocking=True)
    res_8126 = read_res_9430[np.int32(0)]
    y_8127 = slt32(np.int32(7), x_8098)
    assert y_8127, 'At core_shell_parallelepiped_futhark.fut:29:20-29:20: index out of bounds'
    read_res_9431 = np.empty(np.int32(1), dtype=ct.c_double)
    cl.enqueue_copy(self.queue, read_res_9431, values_mem_9354,
                    device_offset=np.long(np.int32(72)), is_blocking=True)
    res_8129 = read_res_9431[np.int32(0)]
    y_8130 = slt32(np.int32(8), x_8098)
    assert y_8130, 'At core_shell_parallelepiped_futhark.fut:30:23-30:23: index out of bounds'
    read_res_9432 = np.empty(np.int32(1), dtype=ct.c_double)
    cl.enqueue_copy(self.queue, read_res_9432, values_mem_9354,
                    device_offset=np.long(np.int32(80)), is_blocking=True)
    res_8132 = read_res_9432[np.int32(0)]
    y_8133 = slt32(np.int32(9), x_8098)
    assert y_8133, 'At core_shell_parallelepiped_futhark.fut:31:23-31:23: index out of bounds'
    read_res_9433 = np.empty(np.int32(1), dtype=ct.c_double)
    cl.enqueue_copy(self.queue, read_res_9433, values_mem_9354,
                    device_offset=np.long(np.int32(88)), is_blocking=True)
    res_8135 = read_res_9433[np.int32(0)]
    res_8136 = (res_8123 / res_8126)
    res_8137 = (res_8129 / res_8126)
    res_8138 = (np.float64(2.0) * res_8132)
    res_8139 = (res_8136 + res_8138)
    res_8140 = (res_8139 / res_8126)
    res_8141 = (np.float64(2.0) * res_8135)
    res_8142 = (res_8136 + res_8141)
    res_8143 = (res_8142 / res_8126)
    res_8144 = (res_8123 * res_8126)
    res_8145 = (res_8144 * res_8129)
    res_8146 = (res_8138 * res_8126)
    res_8147 = (res_8146 * res_8129)
    res_8148 = (np.float64(2.0) * res_8123)
    res_8149 = (res_8148 * res_8135)
    res_8150 = (res_8149 * res_8129)
    res_8151 = (res_8111 - res_8120)
    res_8152 = (res_8114 - res_8120)
    res_8153 = (res_8117 - res_8120)
    res_8154 = (res_8152 * res_8147)
    res_8155 = (res_8153 * res_8150)
    res_8156 = (res_8151 * res_8145)
    res_8157 = (res_8156 - res_8154)
    res_8158 = (res_8157 - res_8155)
    mem_9359 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         np.long(np.long(np.int64(608)) if (np.int64(608) > np.int32(0)) else np.int32(1)))
    static_array_9397 = self.static_array_9397
    if ((np.int32(76) * np.int32(8)) != np.int32(0)):
      cl.enqueue_copy(self.queue, mem_9359, static_array_9397,
                      dest_offset=np.long(np.int64(0)),
                      src_offset=np.long(np.int64(0)),
                      byte_count=np.long((np.int32(76) * np.int32(8))))
    if synchronous:
      self.queue.finish()
    mem_9362 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         np.long(np.long(np.int64(608)) if (np.int64(608) > np.int32(0)) else np.int32(1)))
    static_array_9398 = self.static_array_9398
    if ((np.int32(76) * np.int32(8)) != np.int32(0)):
      cl.enqueue_copy(self.queue, mem_9362, static_array_9398,
                      dest_offset=np.long(np.int64(0)),
                      src_offset=np.long(np.int64(0)),
                      byte_count=np.long((np.int32(76) * np.int32(8))))
    if synchronous:
      self.queue.finish()
    cond_8378 = (num_active_8087 == np.int32(0))
    x_8379 = abs(nq_8088)
    if cond_8378:
      branch_ctx_8380 = x_8379
    else:
      branch_ctx_8380 = nq_8088
    bounds_invalid_upwards_8381 = slt32(nq_8088, np.int32(0))
    binop_x_9370 = sext_i32_i64(sizze_8085)
    bytes_9369 = (binop_x_9370 * np.int64(8))
    mem_9371 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         np.long(np.long(bytes_9369) if (bytes_9369 > np.int32(0)) else np.int32(1)))
    binop_x_9379 = sext_i32_i64(nq_8088)
    bytes_9378 = (binop_x_9379 * np.int64(8))
    mem_9380 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         np.long(np.long(bytes_9378) if (bytes_9378 > np.int32(0)) else np.int32(1)))
    group_sizze_9119 = self.group_size
    binop_y_9364 = sext_i32_i64(group_sizze_9119)
    bytes_9363 = (np.int64(8) * binop_y_9364)
    group_sizze_9242 = self.group_size
    binop_y_9373 = sext_i32_i64(group_sizze_9242)
    bytes_9372 = (np.int64(8) * binop_y_9373)
    if cond_8378:
      y_8398 = slt32(np.int32(10), x_8098)
      assert y_8398, 'At core_shell_parallelepiped_futhark.fut:147:23-147:23: index out of bounds'
      read_res_9436 = np.empty(np.int32(1), dtype=ct.c_double)
      cl.enqueue_copy(self.queue, read_res_9436, values_mem_9354,
                      device_offset=np.long(np.int32(96)), is_blocking=True)
      res_8400 = read_res_9436[np.int32(0)]
      res_8406 = (res_8145 + res_8147)
      res_8408 = (res_8141 * res_8123)
      res_8409 = (res_8408 * res_8129)
      res_8410 = (res_8406 + res_8409)
      res_8411 = (np.float64(2.0) * res_8400)
      res_8412 = (res_8411 * res_8123)
      res_8413 = (res_8412 * res_8126)
      res_8414 = (res_8410 + res_8413)
      res_8415 = (res_8414 == np.float64(0.0))
      if res_8415:
        y_8417 = slt32(np.int32(0), sizze_8084)
        assert y_8417, 'At ./header/for_iq.fut:22:59-22:59: index out of bounds'
        read_res_9437 = np.empty(np.int32(1), dtype=ct.c_double)
        cl.enqueue_copy(self.queue, read_res_9437, values_mem_9354,
                        device_offset=np.long(np.int32(0)), is_blocking=True)
        res_8419 = read_res_9437[np.int32(0)]
        res_8416 = res_8419
      else:
        y_8420 = slt32(np.int32(0), sizze_8084)
        assert y_8420, 'At ./header/for_iq.fut:22:74-22:74: index out of bounds'
        read_res_9438 = np.empty(np.int32(1), dtype=ct.c_double)
        cl.enqueue_copy(self.queue, read_res_9438, values_mem_9354,
                        device_offset=np.long(np.int32(0)), is_blocking=True)
        arg_8422 = read_res_9438[np.int32(0)]
        res_8423 = (arg_8422 / res_8414)
        res_8416 = res_8423
      y_9120 = (group_sizze_9119 - np.int32(1))
      x_9121 = (sizze_8085 + y_9120)
      num_groups_9122 = squot32(x_9121, group_sizze_9119)
      num_threads_9123 = (num_groups_9122 * group_sizze_9119)
      if ((np.int32(1) * (num_groups_9122 * group_sizze_9119)) != np.int32(0)):
        self.map_kernel_9124_var.set_args(cl.LocalMemory(np.long(bytes_9363)),
                                          cl.LocalMemory(np.long(bytes_9363)),
                                          np.int32(sizze_8085),
                                          np.float64(res_8096),
                                          np.float64(res_8126),
                                          np.float64(res_8136),
                                          np.float64(res_8137),
                                          np.float64(res_8140),
                                          np.float64(res_8143),
                                          np.float64(res_8154),
                                          np.float64(res_8155),
                                          np.float64(res_8158),
                                          np.float64(res_8416),
                                          q_input_mem_9356, mem_9359, mem_9362,
                                          mem_9371)
        cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_9124_var,
                                   (np.long((num_groups_9122 * group_sizze_9119)),),
                                   (np.long(group_sizze_9119),))
        if synchronous:
          self.queue.finish()
      empty_slice_8644 = (x_8379 == np.int32(0))
      m_8645 = (x_8379 - np.int32(1))
      zzero_leq_i_p_m_t_s_8646 = sle32(np.int32(0), m_8645)
      i_p_m_t_s_leq_w_8647 = slt32(m_8645, sizze_8085)
      i_lte_j_8648 = sle32(np.int32(0), nq_8088)
      y_8649 = (zzero_leq_i_p_m_t_s_8646 and i_p_m_t_s_leq_w_8647)
      y_8650 = (i_lte_j_8648 and y_8649)
      ok_or_empty_8651 = (empty_slice_8644 or y_8650)
      assert ok_or_empty_8651, 'At ./header/for_iq.fut:27:12-27:12: slice out of bounds'
      res_mem_9382 = mem_9371
      res_mem_sizze_9381 = bytes_9369
    else:
      not_p_8655 = not(bounds_invalid_upwards_8381)
      assert not_p_8655, 'At ./futlib/array.fut:42:5-42:5: shape of function result does not match shapes in return type'
      res_8657 = (nq_8088 + np.int32(1))
      res_8658 = sdiv32(np.int32(1000000), res_8657)
      res_8659 = smax32(res_8658, np.int32(1))
      res_8661 = (cutoff_8093 < np.float64(1.0))
      r_8662 = np.float64(0.0)
      x_8667 = (np.int32(2) * res_8659)
      cond_8668 = slt32(x_8667, call_details_num_evals_8089)
      if cond_8668:
        y_8670 = slt32(np.int32(0), sizze_8084)
        assert y_8670, 'At ./header/for_iq.fut:47:57-47:57: index out of bounds'
        read_res_9439 = np.empty(np.int32(1), dtype=ct.c_double)
        cl.enqueue_copy(self.queue, read_res_9439, values_mem_9354,
                        device_offset=np.long(np.int32(0)), is_blocking=True)
        arg_8672 = read_res_9439[np.int32(0)]
        arg_8673 = (call_details_num_evals_8089 - res_8659)
        res_8674 = sitofp_i32_f64(arg_8673)
        res_8675 = (arg_8672 / res_8674)
        res_8669 = res_8675
      else:
        y_8676 = slt32(np.int32(0), sizze_8084)
        assert y_8676, 'At ./header/for_iq.fut:47:125-47:125: index out of bounds'
        read_res_9440 = np.empty(np.int32(1), dtype=ct.c_double)
        cl.enqueue_copy(self.queue, read_res_9440, values_mem_9354,
                        device_offset=np.long(np.int32(0)), is_blocking=True)
        res_8678 = read_res_9440[np.int32(0)]
        res_8669 = res_8678
      y_9243 = (group_sizze_9242 - np.int32(1))
      x_9244 = (nq_8088 + y_9243)
      num_groups_9245 = squot32(x_9244, group_sizze_9242)
      num_threads_9246 = (num_groups_9245 * group_sizze_9242)
      if ((np.int32(1) * (num_groups_9245 * group_sizze_9242)) != np.int32(0)):
        self.map_kernel_9247_var.set_args(cl.LocalMemory(np.long(bytes_9372)),
                                          cl.LocalMemory(np.long(bytes_9372)),
                                          np.int32(nq_8088),
                                          np.float64(res_8096),
                                          np.float64(res_8126),
                                          np.float64(res_8136),
                                          np.float64(res_8137),
                                          np.float64(res_8140),
                                          np.float64(res_8143),
                                          np.float64(res_8154),
                                          np.float64(res_8155),
                                          np.float64(res_8158),
                                          np.byte(res_8661),
                                          np.float64(res_8669),
                                          q_input_mem_9356, mem_9359, mem_9362,
                                          mem_9380)
        cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_9247_var,
                                   (np.long((num_groups_9245 * group_sizze_9242)),),
                                   (np.long(group_sizze_9242),))
        if synchronous:
          self.queue.finish()
      res_mem_9382 = mem_9380
      res_mem_sizze_9381 = bytes_9378
    out_mem_9394 = res_mem_9382
    out_arrsizze_9396 = branch_ctx_8380
    out_memsizze_9395 = res_mem_sizze_9381
    return (out_memsizze_9395, out_mem_9394, out_arrsizze_9396)
  def kernel_float64_2d(self, num_pars_7634_ext, num_active_7635_ext,
                        nq_7636_ext, call_details_num_evals_7637_ext,
                        call_details_buffer_mem_9352_ext, values_mem_9354_ext,
                        qx_input_mem_9356_ext, qy_input_mem_9358_ext,
                        cutoff_7642_ext):
    num_pars_7634 = np.int32(ct.c_int32(num_pars_7634_ext))
    num_active_7635 = np.int32(ct.c_int32(num_active_7635_ext))
    nq_7636 = np.int32(ct.c_int32(nq_7636_ext))
    call_details_num_evals_7637 = np.int32(ct.c_int32(call_details_num_evals_7637_ext))
    assert ((type(call_details_buffer_mem_9352_ext) in [np.ndarray,
                                                        cl.array.Array]) and (call_details_buffer_mem_9352_ext.dtype == np.int32)), 'Parameter has unexpected type'
    try:
      assert (sizze_7630 == np.int32(call_details_buffer_mem_9352_ext.shape[np.int32(0)])), 'variant dimension wrong'
    except NameError as e:
      sizze_7630 = np.int32(call_details_buffer_mem_9352_ext.shape[np.int32(0)])
    call_details_buffer_mem_sizze_9351 = np.int64(call_details_buffer_mem_9352_ext.nbytes)
    if (type(call_details_buffer_mem_9352_ext) == cl.array.Array):
      call_details_buffer_mem_9352 = call_details_buffer_mem_9352_ext.data
    else:
      call_details_buffer_mem_9352 = cl.Buffer(self.ctx,
                                               cl.mem_flags.READ_WRITE,
                                               np.long(np.long(call_details_buffer_mem_sizze_9351) if (call_details_buffer_mem_sizze_9351 > np.int32(0)) else np.int32(1)))
      if (call_details_buffer_mem_sizze_9351 != np.int32(0)):
        cl.enqueue_copy(self.queue, call_details_buffer_mem_9352,
                        normaliseArray(call_details_buffer_mem_9352_ext),
                        is_blocking=synchronous)
    assert ((type(values_mem_9354_ext) in [np.ndarray,
                                           cl.array.Array]) and (values_mem_9354_ext.dtype == np.float64)), 'Parameter has unexpected type'
    try:
      assert (sizze_7631 == np.int32(values_mem_9354_ext.shape[np.int32(0)])), 'variant dimension wrong'
    except NameError as e:
      sizze_7631 = np.int32(values_mem_9354_ext.shape[np.int32(0)])
    values_mem_sizze_9353 = np.int64(values_mem_9354_ext.nbytes)
    if (type(values_mem_9354_ext) == cl.array.Array):
      values_mem_9354 = values_mem_9354_ext.data
    else:
      values_mem_9354 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                  np.long(np.long(values_mem_sizze_9353) if (values_mem_sizze_9353 > np.int32(0)) else np.int32(1)))
      if (values_mem_sizze_9353 != np.int32(0)):
        cl.enqueue_copy(self.queue, values_mem_9354,
                        normaliseArray(values_mem_9354_ext),
                        is_blocking=synchronous)
    assert ((type(qx_input_mem_9356_ext) in [np.ndarray,
                                             cl.array.Array]) and (qx_input_mem_9356_ext.dtype == np.float64)), 'Parameter has unexpected type'
    try:
      assert (sizze_7632 == np.int32(qx_input_mem_9356_ext.shape[np.int32(0)])), 'variant dimension wrong'
    except NameError as e:
      sizze_7632 = np.int32(qx_input_mem_9356_ext.shape[np.int32(0)])
    qx_input_mem_sizze_9355 = np.int64(qx_input_mem_9356_ext.nbytes)
    if (type(qx_input_mem_9356_ext) == cl.array.Array):
      qx_input_mem_9356 = qx_input_mem_9356_ext.data
    else:
      qx_input_mem_9356 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                    np.long(np.long(qx_input_mem_sizze_9355) if (qx_input_mem_sizze_9355 > np.int32(0)) else np.int32(1)))
      if (qx_input_mem_sizze_9355 != np.int32(0)):
        cl.enqueue_copy(self.queue, qx_input_mem_9356,
                        normaliseArray(qx_input_mem_9356_ext),
                        is_blocking=synchronous)
    assert ((type(qy_input_mem_9358_ext) in [np.ndarray,
                                             cl.array.Array]) and (qy_input_mem_9358_ext.dtype == np.float64)), 'Parameter has unexpected type'
    try:
      assert (sizze_7633 == np.int32(qy_input_mem_9358_ext.shape[np.int32(0)])), 'variant dimension wrong'
    except NameError as e:
      sizze_7633 = np.int32(qy_input_mem_9358_ext.shape[np.int32(0)])
    qy_input_mem_sizze_9357 = np.int64(qy_input_mem_9358_ext.nbytes)
    if (type(qy_input_mem_9358_ext) == cl.array.Array):
      qy_input_mem_9358 = qy_input_mem_9358_ext.data
    else:
      qy_input_mem_9358 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                    np.long(np.long(qy_input_mem_sizze_9357) if (qy_input_mem_sizze_9357 > np.int32(0)) else np.int32(1)))
      if (qy_input_mem_sizze_9357 != np.int32(0)):
        cl.enqueue_copy(self.queue, qy_input_mem_9358,
                        normaliseArray(qy_input_mem_9358_ext),
                        is_blocking=synchronous)
    cutoff_7642 = np.float64(ct.c_double(cutoff_7642_ext))
    (out_memsizze_9386, out_mem_9385,
     out_arrsizze_9387) = self.futhark_kernel_float64_2d(call_details_buffer_mem_sizze_9351,
                                                         values_mem_sizze_9353,
                                                         qx_input_mem_sizze_9355,
                                                         qy_input_mem_sizze_9357,
                                                         call_details_buffer_mem_9352,
                                                         values_mem_9354,
                                                         qx_input_mem_9356,
                                                         qy_input_mem_9358,
                                                         sizze_7630, sizze_7631,
                                                         sizze_7632, sizze_7633,
                                                         num_pars_7634,
                                                         num_active_7635,
                                                         nq_7636,
                                                         call_details_num_evals_7637,
                                                         cutoff_7642)
    return cl.array.Array(self.queue, (out_arrsizze_9387,), ct.c_double,
                          data=out_mem_9385)
  def kernel_float64(self, num_pars_8086_ext, num_active_8087_ext, nq_8088_ext,
                     call_details_num_evals_8089_ext,
                     call_details_buffer_mem_9352_ext, values_mem_9354_ext,
                     q_input_mem_9356_ext, cutoff_8093_ext):
    num_pars_8086 = np.int32(ct.c_int32(num_pars_8086_ext))
    num_active_8087 = np.int32(ct.c_int32(num_active_8087_ext))
    nq_8088 = np.int32(ct.c_int32(nq_8088_ext))
    call_details_num_evals_8089 = np.int32(ct.c_int32(call_details_num_evals_8089_ext))
    assert ((type(call_details_buffer_mem_9352_ext) in [np.ndarray,
                                                        cl.array.Array]) and (call_details_buffer_mem_9352_ext.dtype == np.int32)), 'Parameter has unexpected type'
    try:
      assert (sizze_8083 == np.int32(call_details_buffer_mem_9352_ext.shape[np.int32(0)])), 'variant dimension wrong'
    except NameError as e:
      sizze_8083 = np.int32(call_details_buffer_mem_9352_ext.shape[np.int32(0)])
    call_details_buffer_mem_sizze_9351 = np.int64(call_details_buffer_mem_9352_ext.nbytes)
    if (type(call_details_buffer_mem_9352_ext) == cl.array.Array):
      call_details_buffer_mem_9352 = call_details_buffer_mem_9352_ext.data
    else:
      call_details_buffer_mem_9352 = cl.Buffer(self.ctx,
                                               cl.mem_flags.READ_WRITE,
                                               np.long(np.long(call_details_buffer_mem_sizze_9351) if (call_details_buffer_mem_sizze_9351 > np.int32(0)) else np.int32(1)))
      if (call_details_buffer_mem_sizze_9351 != np.int32(0)):
        cl.enqueue_copy(self.queue, call_details_buffer_mem_9352,
                        normaliseArray(call_details_buffer_mem_9352_ext),
                        is_blocking=synchronous)
    assert ((type(values_mem_9354_ext) in [np.ndarray,
                                           cl.array.Array]) and (values_mem_9354_ext.dtype == np.float64)), 'Parameter has unexpected type'
    try:
      assert (sizze_8084 == np.int32(values_mem_9354_ext.shape[np.int32(0)])), 'variant dimension wrong'
    except NameError as e:
      sizze_8084 = np.int32(values_mem_9354_ext.shape[np.int32(0)])
    values_mem_sizze_9353 = np.int64(values_mem_9354_ext.nbytes)
    if (type(values_mem_9354_ext) == cl.array.Array):
      values_mem_9354 = values_mem_9354_ext.data
    else:
      values_mem_9354 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                  np.long(np.long(values_mem_sizze_9353) if (values_mem_sizze_9353 > np.int32(0)) else np.int32(1)))
      if (values_mem_sizze_9353 != np.int32(0)):
        cl.enqueue_copy(self.queue, values_mem_9354,
                        normaliseArray(values_mem_9354_ext),
                        is_blocking=synchronous)
    assert ((type(q_input_mem_9356_ext) in [np.ndarray,
                                            cl.array.Array]) and (q_input_mem_9356_ext.dtype == np.float64)), 'Parameter has unexpected type'
    try:
      assert (sizze_8085 == np.int32(q_input_mem_9356_ext.shape[np.int32(0)])), 'variant dimension wrong'
    except NameError as e:
      sizze_8085 = np.int32(q_input_mem_9356_ext.shape[np.int32(0)])
    q_input_mem_sizze_9355 = np.int64(q_input_mem_9356_ext.nbytes)
    if (type(q_input_mem_9356_ext) == cl.array.Array):
      q_input_mem_9356 = q_input_mem_9356_ext.data
    else:
      q_input_mem_9356 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                   np.long(np.long(q_input_mem_sizze_9355) if (q_input_mem_sizze_9355 > np.int32(0)) else np.int32(1)))
      if (q_input_mem_sizze_9355 != np.int32(0)):
        cl.enqueue_copy(self.queue, q_input_mem_9356,
                        normaliseArray(q_input_mem_9356_ext),
                        is_blocking=synchronous)
    cutoff_8093 = np.float64(ct.c_double(cutoff_8093_ext))
    (out_memsizze_9395, out_mem_9394,
     out_arrsizze_9396) = self.futhark_kernel_float64(call_details_buffer_mem_sizze_9351,
                                                      values_mem_sizze_9353,
                                                      q_input_mem_sizze_9355,
                                                      call_details_buffer_mem_9352,
                                                      values_mem_9354,
                                                      q_input_mem_9356,
                                                      sizze_8083, sizze_8084,
                                                      sizze_8085, num_pars_8086,
                                                      num_active_8087, nq_8088,
                                                      call_details_num_evals_8089,
                                                      cutoff_8093)
    return cl.array.Array(self.queue, (out_arrsizze_9396,), ct.c_double,
                          data=out_mem_9394)