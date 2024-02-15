/*
MIT License

Copyright (c) 2023 user140242

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/


///     This is a implementation of the bit wheel double segmented sieve 
///     multi-threaded with OpenMP (compile with -fopenmp option)
///     with max base wheel size choice  30 , 210 , 2310 
///     Alternative solution and with the possibility of choosing n_start - v 1_1 - user140242

#include <iostream>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <stdint.h>
#include <string.h>
#include <omp.h>
#include <chrono>

const uint64_t n_PB_max = 5;
const uint64_t Primes_Base[n_PB_max] = {2,3,5,7,11};

const uint64_t del_bit[8] =
{
  ~(1ull << 7),~(1ull << 6),~(1ull << 5),~(1ull << 4), ~(1ull << 3),~(1ull << 2),~(1ull << 1),~(1ull << 0)
};

const uint64_t bit_count[256] =
{
  0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
  1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
  1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
  2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
  1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
  2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
  2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
  3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8
};
    
const uint64_t bit_pos[256][8] =
{
  {0,0,0,0,0,0,0,0},{7,0,0,0,0,0,0,0},{6,0,0,0,0,0,0,0},{6,7,0,0,0,0,0,0},{5,0,0,0,0,0,0,0},{5,7,0,0,0,0,0,0},{5,6,0,0,0,0,0,0},{5,6,7,0,0,0,0,0},
  {4,0,0,0,0,0,0,0},{4,7,0,0,0,0,0,0},{4,6,0,0,0,0,0,0},{4,6,7,0,0,0,0,0},{4,5,0,0,0,0,0,0},{4,5,7,0,0,0,0,0},{4,5,6,0,0,0,0,0},{4,5,6,7,0,0,0,0},
  {3,0,0,0,0,0,0,0},{3,7,0,0,0,0,0,0},{3,6,0,0,0,0,0,0},{3,6,7,0,0,0,0,0},{3,5,0,0,0,0,0,0},{3,5,7,0,0,0,0,0},{3,5,6,0,0,0,0,0},{3,5,6,7,0,0,0,0},
  {3,4,0,0,0,0,0,0},{3,4,7,0,0,0,0,0},{3,4,6,0,0,0,0,0},{3,4,6,7,0,0,0,0},{3,4,5,0,0,0,0,0},{3,4,5,7,0,0,0,0},{3,4,5,6,0,0,0,0},{3,4,5,6,7,0,0,0},
  {2,0,0,0,0,0,0,0},{2,7,0,0,0,0,0,0},{2,6,0,0,0,0,0,0},{2,6,7,0,0,0,0,0},{2,5,0,0,0,0,0,0},{2,5,7,0,0,0,0,0},{2,5,6,0,0,0,0,0},{2,5,6,7,0,0,0,0},
  {2,4,0,0,0,0,0,0},{2,4,7,0,0,0,0,0},{2,4,6,0,0,0,0,0},{2,4,6,7,0,0,0,0},{2,4,5,0,0,0,0,0},{2,4,5,7,0,0,0,0},{2,4,5,6,0,0,0,0},{2,4,5,6,7,0,0,0},
  {2,3,0,0,0,0,0,0},{2,3,7,0,0,0,0,0},{2,3,6,0,0,0,0,0},{2,3,6,7,0,0,0,0},{2,3,5,0,0,0,0,0},{2,3,5,7,0,0,0,0},{2,3,5,6,0,0,0,0},{2,3,5,6,7,0,0,0},
  {2,3,4,0,0,0,0,0},{2,3,4,7,0,0,0,0},{2,3,4,6,0,0,0,0},{2,3,4,6,7,0,0,0},{2,3,4,5,0,0,0,0},{2,3,4,5,7,0,0,0},{2,3,4,5,6,0,0,0},{2,3,4,5,6,7,0,0},
  {1,0,0,0,0,0,0,0},{1,7,0,0,0,0,0,0},{1,6,0,0,0,0,0,0},{1,6,7,0,0,0,0,0},{1,5,0,0,0,0,0,0},{1,5,7,0,0,0,0,0},{1,5,6,0,0,0,0,0},{1,5,6,7,0,0,0,0},
  {1,4,0,0,0,0,0,0},{1,4,7,0,0,0,0,0},{1,4,6,0,0,0,0,0},{1,4,6,7,0,0,0,0},{1,4,5,0,0,0,0,0},{1,4,5,7,0,0,0,0},{1,4,5,6,0,0,0,0},{1,4,5,6,7,0,0,0},
  {1,3,0,0,0,0,0,0},{1,3,7,0,0,0,0,0},{1,3,6,0,0,0,0,0},{1,3,6,7,0,0,0,0},{1,3,5,0,0,0,0,0},{1,3,5,7,0,0,0,0},{1,3,5,6,0,0,0,0},{1,3,5,6,7,0,0,0},
  {1,3,4,0,0,0,0,0},{1,3,4,7,0,0,0,0},{1,3,4,6,0,0,0,0},{1,3,4,6,7,0,0,0},{1,3,4,5,0,0,0,0},{1,3,4,5,7,0,0,0},{1,3,4,5,6,0,0,0},{1,3,4,5,6,7,0,0},
  {1,2,0,0,0,0,0,0},{1,2,7,0,0,0,0,0},{1,2,6,0,0,0,0,0},{1,2,6,7,0,0,0,0},{1,2,5,0,0,0,0,0},{1,2,5,7,0,0,0,0},{1,2,5,6,0,0,0,0},{1,2,5,6,7,0,0,0},
  {1,2,4,0,0,0,0,0},{1,2,4,7,0,0,0,0},{1,2,4,6,0,0,0,0},{1,2,4,6,7,0,0,0},{1,2,4,5,0,0,0,0},{1,2,4,5,7,0,0,0},{1,2,4,5,6,0,0,0},{1,2,4,5,6,7,0,0},
  {1,2,3,0,0,0,0,0},{1,2,3,7,0,0,0,0},{1,2,3,6,0,0,0,0},{1,2,3,6,7,0,0,0},{1,2,3,5,0,0,0,0},{1,2,3,5,7,0,0,0},{1,2,3,5,6,0,0,0},{1,2,3,5,6,7,0,0},
  {1,2,3,4,0,0,0,0},{1,2,3,4,7,0,0,0},{1,2,3,4,6,0,0,0},{1,2,3,4,6,7,0,0},{1,2,3,4,5,0,0,0},{1,2,3,4,5,7,0,0},{1,2,3,4,5,6,0,0},{1,2,3,4,5,6,7,0},
  {0,0,0,0,0,0,0,0},{0,7,0,0,0,0,0,0},{0,6,0,0,0,0,0,0},{0,6,7,0,0,0,0,0},{0,5,0,0,0,0,0,0},{0,5,7,0,0,0,0,0},{0,5,6,0,0,0,0,0},{0,5,6,7,0,0,0,0},
  {0,4,0,0,0,0,0,0},{0,4,7,0,0,0,0,0},{0,4,6,0,0,0,0,0},{0,4,6,7,0,0,0,0},{0,4,5,0,0,0,0,0},{0,4,5,7,0,0,0,0},{0,4,5,6,0,0,0,0},{0,4,5,6,7,0,0,0},
  {0,3,0,0,0,0,0,0},{0,3,7,0,0,0,0,0},{0,3,6,0,0,0,0,0},{0,3,6,7,0,0,0,0},{0,3,5,0,0,0,0,0},{0,3,5,7,0,0,0,0},{0,3,5,6,0,0,0,0},{0,3,5,6,7,0,0,0},
  {0,3,4,0,0,0,0,0},{0,3,4,7,0,0,0,0},{0,3,4,6,0,0,0,0},{0,3,4,6,7,0,0,0},{0,3,4,5,0,0,0,0},{0,3,4,5,7,0,0,0},{0,3,4,5,6,0,0,0},{0,3,4,5,6,7,0,0},
  {0,2,0,0,0,0,0,0},{0,2,7,0,0,0,0,0},{0,2,6,0,0,0,0,0},{0,2,6,7,0,0,0,0},{0,2,5,0,0,0,0,0},{0,2,5,7,0,0,0,0},{0,2,5,6,0,0,0,0},{0,2,5,6,7,0,0,0},
  {0,2,4,0,0,0,0,0},{0,2,4,7,0,0,0,0},{0,2,4,6,0,0,0,0},{0,2,4,6,7,0,0,0},{0,2,4,5,0,0,0,0},{0,2,4,5,7,0,0,0},{0,2,4,5,6,0,0,0},{0,2,4,5,6,7,0,0},
  {0,2,3,0,0,0,0,0},{0,2,3,7,0,0,0,0},{0,2,3,6,0,0,0,0},{0,2,3,6,7,0,0,0},{0,2,3,5,0,0,0,0},{0,2,3,5,7,0,0,0},{0,2,3,5,6,0,0,0},{0,2,3,5,6,7,0,0},
  {0,2,3,4,0,0,0,0},{0,2,3,4,7,0,0,0},{0,2,3,4,6,0,0,0},{0,2,3,4,6,7,0,0},{0,2,3,4,5,0,0,0},{0,2,3,4,5,7,0,0},{0,2,3,4,5,6,0,0},{0,2,3,4,5,6,7,0},
  {0,1,0,0,0,0,0,0},{0,1,7,0,0,0,0,0},{0,1,6,0,0,0,0,0},{0,1,6,7,0,0,0,0},{0,1,5,0,0,0,0,0},{0,1,5,7,0,0,0,0},{0,1,5,6,0,0,0,0},{0,1,5,6,7,0,0,0},
  {0,1,4,0,0,0,0,0},{0,1,4,7,0,0,0,0},{0,1,4,6,0,0,0,0},{0,1,4,6,7,0,0,0},{0,1,4,5,0,0,0,0},{0,1,4,5,7,0,0,0},{0,1,4,5,6,0,0,0},{0,1,4,5,6,7,0,0},
  {0,1,3,0,0,0,0,0},{0,1,3,7,0,0,0,0},{0,1,3,6,0,0,0,0},{0,1,3,6,7,0,0,0},{0,1,3,5,0,0,0,0},{0,1,3,5,7,0,0,0},{0,1,3,5,6,0,0,0},{0,1,3,5,6,7,0,0},
  {0,1,3,4,0,0,0,0},{0,1,3,4,7,0,0,0},{0,1,3,4,6,0,0,0},{0,1,3,4,6,7,0,0},{0,1,3,4,5,0,0,0},{0,1,3,4,5,7,0,0},{0,1,3,4,5,6,0,0},{0,1,3,4,5,6,7,0},
  {0,1,2,0,0,0,0,0},{0,1,2,7,0,0,0,0},{0,1,2,6,0,0,0,0},{0,1,2,6,7,0,0,0},{0,1,2,5,0,0,0,0},{0,1,2,5,7,0,0,0},{0,1,2,5,6,0,0,0},{0,1,2,5,6,7,0,0},
  {0,1,2,4,0,0,0,0},{0,1,2,4,7,0,0,0},{0,1,2,4,6,0,0,0},{0,1,2,4,6,7,0,0},{0,1,2,4,5,0,0,0},{0,1,2,4,5,7,0,0},{0,1,2,4,5,6,0,0},{0,1,2,4,5,6,7,0},
  {0,1,2,3,0,0,0,0},{0,1,2,3,7,0,0,0},{0,1,2,3,6,0,0,0},{0,1,2,3,6,7,0,0},{0,1,2,3,5,0,0,0},{0,1,2,3,5,7,0,0},{0,1,2,3,5,6,0,0},{0,1,2,3,5,6,7,0},
  {0,1,2,3,4,0,0,0},{0,1,2,3,4,7,0,0},{0,1,2,3,4,6,0,0},{0,1,2,3,4,6,7,0},{0,1,2,3,4,5,0,0},{0,1,2,3,4,5,7,0},{0,1,2,3,4,5,6,0},{0,1,2,3,4,5,6,7}
};

uint64_t double_segmented_bit_sieve_wheel_seg(uint64_t k_low_start, uint64_t k_low_stop, uint64_t n_i_mod_bW , uint64_t k_i , uint64_t n_f_mod_bW, uint64_t k_end , uint64_t bW, std::vector<uint64_t> RW, std::vector<uint64_t> C1, std::vector<uint64_t> RW_i, std::vector<uint8_t> Segment_0_0, uint64_t  segment_size_0 , uint64_t sqrt_segments_size , uint64_t p_mask_i, std::vector<uint8_t> Segment_mask, int ck_seg)
{
    //if ck_seg==0    returns the count of prime numbers in first segment from  k_i to k_low_stop
    //if ck_seg==1    returns the count of prime numbers in segments from  k_low_start to k_low_stop
    //if ck_seg==2    returns the count of prime numbers in last segments from  k_low_start to k_end
    
    uint64_t  count_p = 0ull;

    uint64_t nR  = RW.size();
    uint64_t nB = nR / 8; //nB number of byte for residue classes equal to nR=8*nB        
    uint64_t segment_size_b = Segment_mask.size();
    uint64_t segment_size = segment_size_b / nB;
    uint64_t segment_size_0_b = nB * segment_size_0;

    uint64_t segments_size_0_0 = Segment_0_0.size() / nB;        

    uint64_t  p , pb , mb , ip , i , jb , j , jp , kp , k , kmax , kb = 0ull;
    uint8_t  val_B;

    //vector used for segment of size (nB+segment_size_b)=nB*(1+segment_size)       
    std::vector<uint8_t> Segment_t(segment_size_b);
    std::vector<uint8_t> Segment_0_t(segment_size_0_b);
    
    //segment scan
    uint64_t k_low , k_0_low, kmax_0, k1 , kb_low = 0ull;
    for (k_low = k_low_start; k_low < k_low_stop; k_low += segment_size)
    {
        kb_low = k_low * nB;
        //initialize segment    
        for (kb = 0ull; kb < segment_size_b; kb++)
            Segment_t[kb] = Segment_mask[kb];
        //sieve for the segment
        kmax = std::min(sqrt_segments_size , (uint64_t) std::sqrt((k_low + segment_size) / bW) +  (uint64_t)3);
        k_0_low = 0ull;
        while (k_0_low < kmax)
        {
            std::fill(Segment_0_t.begin(), Segment_0_t.end(), 0xff); 
            if (k_0_low >= segments_size_0_0)
            {
                //sieve for the segment Segment_0_t
                kmax_0 = std::min(segments_size_0_0 , (uint64_t) std::sqrt((k_0_low + segment_size_0) / bW) +  (uint64_t)3);
                for(k1 = 0ull; k1 < kmax_0; k1++)
                {
                    kb = k1 * nB;
                    for (jb = 0ull; jb < nB; jb++)
                    {
                        val_B = Segment_0_0[kb + jb];
                        for (j = 0ull; j < bit_count[val_B]; j++)
                        {
                            jp = bit_pos[val_B][j] + jb * 8ull;
                            p = bW * k1 + RW[jp];
                            for (ip = 0ull; ip < nR; ip++)
                            {
                                mb = k1 * RW_i[ip + jp * nR] + C1[ip + jp * nR];
                                if (mb < k_0_low)
                                    mb = p - ((k_0_low - mb) % p);
                                else
                                    mb -= k_0_low;
                                if (mb == p)
                                    mb = 0ull;
                                if (mb < segment_size_0)
                                {
                                    pb = p * nB;
                                    mb *= nB;
                                    mb += ip / 8;
                                    i = ip % 8;
                                    for (; mb < segment_size_0_b; mb += pb)
                                        Segment_0_t[mb] &= del_bit[i];
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                for (kb = 0ull; kb < segment_size_0_b; kb++)
                    Segment_0_t[kb] = Segment_0_0[kb + k_0_low * nB];
            }
            //sieve for the segment Segment_t
            j = 0ull;
            if (k_0_low == 0ull)
                j = p_mask_i;
            for(k = 0ull; k < std::min(segment_size_0 , kmax - k_0_low); k++)
            {
                kb = k * nB;
                kp = k + k_0_low;
                for (jb = 0ull; jb < nB; jb++)
                {
                    val_B = Segment_0_t[kb + jb];
                    for (; j < bit_count[val_B]; j++)
                    {
                        jp = bit_pos[val_B][j] + jb * 8ull;
                        p = bW * kp + RW[jp];
                        for (ip = 0ull; ip < nR; ip++)
                        {
                            mb = kp * RW_i[ip + jp * nR] + C1[ip + jp * nR];
                            if (mb < k_low)
                                mb = p - ((k_low - mb) % p);
                            else
                                mb -= k_low;
                            if (mb == p)
                                mb = 0ull;
                            if (mb < segment_size)
                            {
                                pb = p * nB;
                                mb *= nB;
                                mb += ip / 8;
                                i = ip % 8;
                                for (; mb < segment_size_b; mb += pb)
                                    Segment_t[mb] &= del_bit[i];
                            }
                        }
                    }
                    j = 0ull;
                }
            }
            k_0_low += segment_size_0;
        }
        //count prime numbers of the segment Segment_t
        if (ck_seg == 1)
        {            
            for (kb = 0ull; kb < segment_size_b; kb++)
                count_p += bit_count[Segment_t[kb]];
        }
        else if (ck_seg == 2)
        {            
            for (kb = kb_low; kb < std::min (kb_low + segment_size_b , nB * k_end); kb++)
                count_p += bit_count[Segment_t[kb - kb_low]];
        }
    }
    if (ck_seg == 0)
    {
        if (k_i >= k_low_start && k_i < k_low_start + segment_size)
        {
            //count prime numbers for k = k_i if k_i > segment_size_0
            for (ip = 0ull; ip < nR; ip++)
                if(Segment_t[(k_i - k_low_start) * nB + ip / 8] & (1 << (7 - (ip % 8))) && RW[ip] >= n_i_mod_bW)
                    count_p++;
        }
        //count prime numbers of the segment
        if (k_i < k_low_start + segment_size - 1ull)
        {
            for (kb = nB + k_i * nB; kb < std::min (nB * k_low_start + segment_size_b, nB * k_end); kb++)
                count_p += bit_count[Segment_t[kb - nB * k_low_start]];
        }
    }
    if (ck_seg != 1 && kb == nB * k_end  && k_end != k_i)
    {
        //count prime numbers for k=k_end
        if (kb - kb_low <= segment_size_b - nB && kb >= kb_low)
            for (ip = 0ull; ip < nR; ip++)
                if(Segment_t[kb - kb_low + ip / 8] & (1 << (7 - (ip % 8))) && RW[ip] < n_f_mod_bW)
                    count_p++;
    }

    return count_p;
}

int gen_base_bW(char *n_str, uint64_t &r, uint64_t &k, uint64_t bW)
{
    int esp_base_N_10 = 18;
    int len_n_str = strlen(n_str);
    if (len_n_str > 32 || len_n_str == 0)
    {
        std::cout << "number too large or wrong" << std::endl;
        return 1;
    }
    else
    {
        char x_t[esp_base_N_10 + 1];
        x_t[0] = n_str[0];
        x_t[1] = '\0';
        uint64_t v_t = std::atoll(x_t);
        if (v_t < 1ull)
        {
            r = 0ull;
            k = 0ull;
        }
        else
        {
            if (len_n_str <= esp_base_N_10)
            {
                r = std::atoll(n_str) % bW;
                k = std::atoll(n_str) / bW;
            }
            else
            {
                char x_t[esp_base_N_10 + 1];
                uint64_t base_N_10 = (uint64_t)std::pow(10 , esp_base_N_10);
                int mod_len = len_n_str % esp_base_N_10;

                //base conversion
                for (int i = mod_len ; i < len_n_str ; i++)
                    x_t[i - mod_len] = n_str[i];
                x_t[esp_base_N_10] = '\0';
                v_t = std::atoll(x_t);
                r = v_t % bW;
                k = v_t / bW;
                for (int i = 0 ; i < mod_len ; i++)
                {
                    x_t[i] = n_str[i];
                }
                x_t[mod_len] = '\0';
                v_t = std::atoll(x_t);
                r += ((base_N_10 % bW) * (v_t % bW)) % bW;
                k += ((base_N_10 % bW) * (v_t % bW)) / bW;
                k += (base_N_10 % bW) * (v_t / bW);
                k += (base_N_10 / bW) * (v_t % bW);
                k += (base_N_10 / bW) * (v_t / bW) * bW;
                k += r / bW;
                r %= bW;
            }
        }
        return 0;
    }
}

uint64_t double_segmented_bit_sieve_wheel_IF_MT(char *n_i, char *n_f, uint64_t max_bW, uint64_t  max_threads)
{
    //returns the count of prime numbers >= n_i and < n_f
    //max_bW is max base wheel size choice max_bW = 30 , 210 , 2310
    //max_threads is maximum number of threads to use

    int ck_val = 1;
    uint64_t k_i = 0ull;
    uint64_t k_end = 0ull;
    uint64_t n_i_mod_bW = 0ull;
    uint64_t n_f_mod_bW = 0ull;

    uint64_t m_segment_size_0 = 1ull; //multiplicity of the minimum segment for segmentation from 0 to sqrt(r_f + bW *k_f)
    uint64_t m_segment_size = 1ull; //multiplicity of the minimum segment for segmentation from r_in + bW *k_in to r_f + bW *k_f
    uint64_t p_mask_i = 4ull;    //0 =< p_mask_i < 8 number primes following those of the basis for pre-sieve vector mask for minimum segment

    uint64_t  count_p = 0ull;

    uint64_t n_PB = 3ull;    
    uint64_t bW = 30ull;
    
    //get bW base wheel equal to p1*p2*...*pn <=max_bW  with n_f=n_PB
    while(n_PB < n_PB_max && (bW * Primes_Base[n_PB] <= max_bW))
    {
        bW *= Primes_Base[n_PB];
        n_PB++;
    }
    
    ck_val = gen_base_bW(n_i, n_i_mod_bW, k_i, bW);
    if (n_i_mod_bW <= 1ull && k_i > 0ull)
    {
        n_i_mod_bW += bW;
        k_i--;
    }
    ck_val += gen_base_bW(n_f, n_f_mod_bW, k_end, bW);
    if (n_f_mod_bW <= 1ull && k_end > 0ull)
    {
        n_f_mod_bW += bW;
        k_end--;
    }
    
    if (ck_val == 0 && k_i == 0)
        for (uint64_t i = 0ull; i < n_PB; i++)
            if (Primes_Base[i] >= n_i_mod_bW)
                if((n_f_mod_bW > Primes_Base[i] && k_end == 0) || k_end > 0)
                    count_p++;

    if ((k_end > 0ull || (k_end == 0ull && Primes_Base[n_PB - 1ull] < n_f_mod_bW)) && (k_end > k_i || (k_end == k_i && n_i_mod_bW < n_f_mod_bW)) && ck_val == 0)
    {
        uint64_t k_sqrt = (uint64_t) std::sqrt(k_end / bW) + 1ull;

        //find reduct residue set modulo bW
        std::vector<char> Remainder_t(bW,true); 
        for (uint64_t i = 0ull; i < n_PB; i++)
            for (uint64_t j = Primes_Base[i]; j < bW; j += Primes_Base[i])
                Remainder_t[j] = false;
        std::vector<uint64_t> RW;
        for (uint64_t j = 2ull; j < bW; j++)
            if (Remainder_t[j] == true)
                RW.push_back(j);
        RW.push_back(1ull + bW);
        uint64_t  nR = RW.size();   //nR=phi(bW)

        //get the matrix constant C1 and RW_i to find the initial multiples of prime numbers in segment
        //C1[i + nR * j]=(RW[j]*RW[j1])/bW and RW_i[i + nR * j]=RW[j1] with (RW[j]*RW[j1])%bW=RW[i]
        std::vector<uint64_t> C1(nR * nR, 0ull);
        std::vector<uint64_t> RW_i(nR * nR, 0ull);
        for (uint64_t i = 0ull; i < nR - 1; i++)
        {
            for (uint64_t j = 0ull; j < nR; j++)
            {
                uint64_t j1 = 0ull;
                while((RW[j] * RW[j1]) % bW != RW[i])
                    j1++;                
                C1[i + nR * j] = (RW[j] * RW[j1]) / bW;
                RW_i[i + nR * j] = RW[j1];
            }
        }
        for (uint64_t j = 0ull; j < nR; j++)
        {
            uint64_t j1 = 0ull;
            while((RW[j] * RW[j1]) % bW != 1)
                j1++;                
            C1[nR - 1 + nR * j] = (RW[j] * RW[j1]) / bW - 1ull;
            RW_i[nR - 1 + nR * j] = RW[j1];
        }

        if (k_end >= 4500000ull)
            m_segment_size = 3ull;
        
        // get segment_size_min=p_(n_PB+1)*p_(n_PB+2)*...*p_(n_PB+p_mask_i)
        uint64_t segment_size_min = 1ull;
        p_mask_i = std::min(p_mask_i, (uint64_t)7); //p_mask_i < 8    
        p_mask_i = std::max(p_mask_i, (uint64_t)0); //p_mask_i >=0
        if (p_mask_i < 2ull)
            segment_size_min = std::max(segment_size_min, (uint64_t)128);            
        for (uint64_t i = 0; i < p_mask_i; i++)
            segment_size_min *= RW[i];
        uint64_t segment_size_0 = m_segment_size_0 * segment_size_min;
        uint64_t sqrt_segments_size = segment_size_0;
        uint64_t segments_size_0_0 = segment_size_0;
        while (sqrt_segments_size <= k_sqrt)
            sqrt_segments_size += segment_size_0;
        while (segments_size_0_0 < (uint64_t) std::sqrt(sqrt_segments_size / bW) + 3ull)
            segments_size_0_0 += segment_size_0;

        uint64_t  nB = nR / 8; //nB number of byte for residue of congruence class equal to nR=8*nB        
        uint64_t segment_size = m_segment_size * segment_size_min;
        uint64_t segment_size_0_b = nB * segment_size_0;
        
        //vector used for the first segment containing prime numbers less than sqrt(sqrt(n_f))
        std::vector<uint8_t> Segment_0_0(nB * segments_size_0_0, 0xff);
 
        uint64_t  p , pb , mb , ip , i , jb , j , jp , k1  , k , kb = 0ull, kb1 = 0ull;
        uint8_t  val_B;
        uint64_t kmax_0 = std::min(segments_size_0_0 , (uint64_t) std::sqrt(segments_size_0_0 / bW) + (uint64_t)3);
        //sieve for the first segment - includes prime numbers < sqrt(sqrt(n_f))
        for (k = 0ull; k  < kmax_0; k++)
        {
            kb = nB * k;
            for (jb = 0ull; jb < nB; jb++)
            {
                val_B = Segment_0_0[kb + jb];
                for (j = 0ull; j < bit_count[val_B]; j++)
                {
                    jp = bit_pos[val_B][j] + jb * 8ull;
                    pb = bW * kb + nB * RW[jp];  // nB * (bW * k + RW[jp]); 
                    for (ip = 0ull; ip < nR; ip++)
                    {
                        mb = ip / 8ull + RW_i[ip + jp * nR] * kb + nB * C1[ip + jp * nR] + k * pb;
                        i = ip % 8;
                        for (; mb < nB * segments_size_0_0; mb += pb)
                            Segment_0_0[mb] &= del_bit[i];
                    }
                }
            }
        }

        if (k_i < sqrt_segments_size)
        {
            std::vector<uint8_t> Segment_0_t(segment_size_0_b);
            uint64_t k_0_low = (k_i / segment_size_0) * segment_size_0;
            //count the prime numbers in the first segment which contains k_i
            while (k_0_low <= k_end && k_0_low < sqrt_segments_size)
            {
                std::fill(Segment_0_t.begin(), Segment_0_t.end(), 0xff);
                if (k_0_low >= segments_size_0_0)
                {
                    //sieve for the segment Segment_0_t
                    kmax_0 = std::min(segments_size_0_0 , (uint64_t) std::sqrt((k_0_low + segment_size_0) / bW) + (uint64_t)3);
                    for(k1 = 0ull; k1 < kmax_0; k1++)
                    {
                        kb = k1 * nB;
                        for (jb = 0ull; jb < nB; jb++)
                        {
                            val_B = Segment_0_0[kb + jb];
                            for (j = 0ull; j < bit_count[val_B]; j++)
                            {
                                jp = bit_pos[val_B][j] + jb * 8ull;
                                p = bW * k1 + RW[jp];
                                for (ip = 0ull; ip < nR; ip++)
                                {
                                    mb = k1 * RW_i[ip + jp * nR] + C1[ip + jp * nR];
                                    if (mb < k_0_low)
                                        mb = p - ((k_0_low - mb) % p);
                                    else
                                        mb -= k_0_low;
                                    if (mb == p)
                                        mb = 0ull;
                                    if (mb < segment_size_0)
                                    {
                                        pb = p * nB;
                                        mb *= nB;
                                        mb += ip / 8;
                                        i = ip % 8;
                                        for (; mb < segment_size_0_b; mb += pb)
                                            Segment_0_t[mb] &= del_bit[i];
                                    }
                                }
                            }
                        }
                    }
                }
                else
                {
                    for (kb = 0ull; kb < segment_size_0_b; kb++)
                        Segment_0_t[kb] = Segment_0_0[kb + k_0_low * nB];
                }
                if (k_0_low == (k_i / segment_size_0) * segment_size_0)
                {
                    //count prime numbers for k = k_i
                    for (ip = 0ull; ip < nR; ip++)
                        if(Segment_0_t[(k_i - k_0_low) * nB + ip / 8] & (1 << (7 - (ip % 8))) && RW[ip] >= n_i_mod_bW)
                            if (k_end > k_i || (RW[ip] < n_f_mod_bW && k_end == k_i))
                                count_p++;
                }
                for (kb1 = nB * (k_i - k_0_low + 1); kb1 < std::min (segment_size_0_b , nB * (k_end - k_0_low)); kb1++)
                    count_p += bit_count[Segment_0_t[kb1]];
                k_0_low += segment_size_0;
            }
            //count prime numbers for k = k_end if k_end < sqrt_segments_size
            if (k_end < sqrt_segments_size && k_end != k_i)
                if (kb1 == nB * (k_end - (k_0_low - segment_size_0)) && kb1 <= segment_size_0_b - nB)
                    for (ip = 0; ip < nR; ip++)
                        if(Segment_0_t[kb1 + ip / 8]& (1 << (7 - (ip % 8))) && RW[ip] < n_f_mod_bW)
                            count_p++;
        }
        
        if (k_end >= sqrt_segments_size) 
        {
            // vector mask pre-sieve multiples of primes bW+RW[j]  with 0<=j<p_mask_i
            std::vector<uint8_t> Segment_mask(nB * segment_size , 0xff);
            for (j = 0ull; j < p_mask_i; j++)
            {
                p = RW[j];
                for (ip = 0ull; ip < nR; ip++)
                {
                    mb = C1[ip + j * nR];
                    if (mb < segment_size_min)
                        mb = p - ((segment_size_min - mb) % p);
                    else
                        mb -= segment_size_min;
                    if (mb == p)
                        mb = 0ull;
                    if (mb < segment_size)
                    {
                        pb = p * nB;
                        mb *= nB;
                        mb += ip / 8;
                        i = ip % 8;                            
                        for (; mb < nB * segment_size; mb += pb)
                            Segment_mask[mb] &= del_bit[i];
                    }
                }
            }

            uint64_t k_start = sqrt_segments_size;
            if (k_i >= sqrt_segments_size)
            {
                //scanning the first segment and count from k_i
                k_start = segment_size_min * (k_i / segment_size_min);
                count_p += double_segmented_bit_sieve_wheel_seg(k_start, std::min (k_start + segment_size , k_end), n_i_mod_bW, k_i ,n_f_mod_bW , k_end, bW, RW, C1, RW_i, Segment_0_0, segment_size_0, sqrt_segments_size, p_mask_i, Segment_mask , 0);
                k_start += segment_size;
            }

            uint64_t range_thread = 0ull;
            uint64_t n_threads = 1ull;
            //multithreaded section
            if (max_threads > 1ull)
            {
                n_threads = omp_get_max_threads();
                n_threads = std::min(n_threads, max_threads);

                while(((k_end - k_start) / n_threads <= segment_size || (k_end - k_start) % n_threads == 0ull) && n_threads > 1ull)
                    n_threads --;
                
                if (n_threads > 1ull)
                {
                    range_thread = (k_end - k_start) / n_threads;
                    range_thread = segment_size * (range_thread / segment_size);
  
                    std::vector<uint64_t> count_p_vector(n_threads, 0ull);

                    #pragma omp parallel num_threads(n_threads)
                    {
                        uint64_t t = omp_get_thread_num();
                        uint64_t thread_start = k_start + t * range_thread;
                        uint64_t thread_stop = thread_start + range_thread;
                        count_p_vector[t] = double_segmented_bit_sieve_wheel_seg(thread_start, thread_stop, n_i_mod_bW, k_i ,n_f_mod_bW, k_end , bW, RW, C1, RW_i, Segment_0_0, segment_size_0, sqrt_segments_size, p_mask_i, Segment_mask , 1);
                    }
                    for (uint64_t t = 0ull; t < n_threads; t++) 
                        count_p += count_p_vector[t];
                }
            }
            //end multithreaded section
            
            //scanning the last remaining segments and count to k_end          
            count_p += double_segmented_bit_sieve_wheel_seg(k_start + n_threads * range_thread, k_end + 1ull ,n_i_mod_bW, k_i ,n_f_mod_bW , k_end, bW, RW, C1, RW_i, Segment_0_0, segment_size_0, sqrt_segments_size, p_mask_i, Segment_mask , 2);
        }
    }

    return count_p;
}

int main()
{
    char n_start[] = "0";
    char n_stop[] = "1000000000";

    uint64_t count = 0;

    auto ti = std::chrono::system_clock::now();
    //double_segmented_bit_sieve_wheel_IF_MT(n_start, n_stop, max_bW, max_threads) 
    //with max base wheel size choice max_bW = 30 , 210 , 2310 and max_threads maximum number of threads to use
    count = double_segmented_bit_sieve_wheel_IF_MT(n_start, n_stop, 210, 1);
    auto tf = std::chrono::system_clock::now();
    
    std::cout << "found " << count << " prime numbers >= " << n_start << " and < " << n_stop << std::endl;
    std::chrono::duration<double> delta_t = tf - ti;
    std::cout << " \n" << "t: " << delta_t.count() << " s" << std::endl; 

    return 0;
}
