///     This is a implementation of the bit wheel segmented sieve 
///     multi-threaded with OpenMP (compile with -fopenmp option)
///     with max base wheel size choice  30 , 210 , 2310 

#include <iostream>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <stdint.h>
#include <omp.h>

const int64_t n_PB_max = 5;
const int64_t Primes_Base[n_PB_max] = {2,3,5,7,11};

const int64_t del_bit[8] =
{
  ~(1 << 0),~(1 << 1),~(1 << 2),~(1 << 3), ~(1 << 4),~(1 << 5),~(1 << 6),~(1 << 7)
};

const int64_t bit_count[256] =
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
    
const int64_t bit_pos[256][8] =
{
  {0,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0},{1,0,0,0,0,0,0,0},{0,1,0,0,0,0,0,0},{2,0,0,0,0,0,0,0},{0,2,0,0,0,0,0,0},{1,2,0,0,0,0,0,0},{0,1,2,0,0,0,0,0},
  {3,0,0,0,0,0,0,0},{0,3,0,0,0,0,0,0},{1,3,0,0,0,0,0,0},{0,1,3,0,0,0,0,0},{2,3,0,0,0,0,0,0},{0,2,3,0,0,0,0,0},{1,2,3,0,0,0,0,0},{0,1,2,3,0,0,0,0},
  {4,0,0,0,0,0,0,0},{0,4,0,0,0,0,0,0},{1,4,0,0,0,0,0,0},{0,1,4,0,0,0,0,0},{2,4,0,0,0,0,0,0},{0,2,4,0,0,0,0,0},{1,2,4,0,0,0,0,0},{0,1,2,4,0,0,0,0},
  {3,4,0,0,0,0,0,0},{0,3,4,0,0,0,0,0},{1,3,4,0,0,0,0,0},{0,1,3,4,0,0,0,0},{2,3,4,0,0,0,0,0},{0,2,3,4,0,0,0,0},{1,2,3,4,0,0,0,0},{0,1,2,3,4,0,0,0},
  {5,0,0,0,0,0,0,0},{0,5,0,0,0,0,0,0},{1,5,0,0,0,0,0,0},{0,1,5,0,0,0,0,0},{2,5,0,0,0,0,0,0},{0,2,5,0,0,0,0,0},{1,2,5,0,0,0,0,0},{0,1,2,5,0,0,0,0},
  {3,5,0,0,0,0,0,0},{0,3,5,0,0,0,0,0},{1,3,5,0,0,0,0,0},{0,1,3,5,0,0,0,0},{2,3,5,0,0,0,0,0},{0,2,3,5,0,0,0,0},{1,2,3,5,0,0,0,0},{0,1,2,3,5,0,0,0},
  {4,5,0,0,0,0,0,0},{0,4,5,0,0,0,0,0},{1,4,5,0,0,0,0,0},{0,1,4,5,0,0,0,0},{2,4,5,0,0,0,0,0},{0,2,4,5,0,0,0,0},{1,2,4,5,0,0,0,0},{0,1,2,4,5,0,0,0},
  {3,4,5,0,0,0,0,0},{0,3,4,5,0,0,0,0},{1,3,4,5,0,0,0,0},{0,1,3,4,5,0,0,0},{2,3,4,5,0,0,0,0},{0,2,3,4,5,0,0,0},{1,2,3,4,5,0,0,0},{0,1,2,3,4,5,0,0},
  {6,0,0,0,0,0,0,0},{0,6,0,0,0,0,0,0},{1,6,0,0,0,0,0,0},{0,1,6,0,0,0,0,0},{2,6,0,0,0,0,0,0},{0,2,6,0,0,0,0,0},{1,2,6,0,0,0,0,0},{0,1,2,6,0,0,0,0},
  {3,6,0,0,0,0,0,0},{0,3,6,0,0,0,0,0},{1,3,6,0,0,0,0,0},{0,1,3,6,0,0,0,0},{2,3,6,0,0,0,0,0},{0,2,3,6,0,0,0,0},{1,2,3,6,0,0,0,0},{0,1,2,3,6,0,0,0},
  {4,6,0,0,0,0,0,0},{0,4,6,0,0,0,0,0},{1,4,6,0,0,0,0,0},{0,1,4,6,0,0,0,0},{2,4,6,0,0,0,0,0},{0,2,4,6,0,0,0,0},{1,2,4,6,0,0,0,0},{0,1,2,4,6,0,0,0},
  {3,4,6,0,0,0,0,0},{0,3,4,6,0,0,0,0},{1,3,4,6,0,0,0,0},{0,1,3,4,6,0,0,0},{2,3,4,6,0,0,0,0},{0,2,3,4,6,0,0,0},{1,2,3,4,6,0,0,0},{0,1,2,3,4,6,0,0},
  {5,6,0,0,0,0,0,0},{0,5,6,0,0,0,0,0},{1,5,6,0,0,0,0,0},{0,1,5,6,0,0,0,0},{2,5,6,0,0,0,0,0},{0,2,5,6,0,0,0,0},{1,2,5,6,0,0,0,0},{0,1,2,5,6,0,0,0},
  {3,5,6,0,0,0,0,0},{0,3,5,6,0,0,0,0},{1,3,5,6,0,0,0,0},{0,1,3,5,6,0,0,0},{2,3,5,6,0,0,0,0},{0,2,3,5,6,0,0,0},{1,2,3,5,6,0,0,0},{0,1,2,3,5,6,0,0},
  {4,5,6,0,0,0,0,0},{0,4,5,6,0,0,0,0},{1,4,5,6,0,0,0,0},{0,1,4,5,6,0,0,0},{2,4,5,6,0,0,0,0},{0,2,4,5,6,0,0,0},{1,2,4,5,6,0,0,0},{0,1,2,4,5,6,0,0},
  {3,4,5,6,0,0,0,0},{0,3,4,5,6,0,0,0},{1,3,4,5,6,0,0,0},{0,1,3,4,5,6,0,0},{2,3,4,5,6,0,0,0},{0,2,3,4,5,6,0,0},{1,2,3,4,5,6,0,0},{0,1,2,3,4,5,6,0},
  {7,0,0,0,0,0,0,0},{0,7,0,0,0,0,0,0},{1,7,0,0,0,0,0,0},{0,1,7,0,0,0,0,0},{2,7,0,0,0,0,0,0},{0,2,7,0,0,0,0,0},{1,2,7,0,0,0,0,0},{0,1,2,7,0,0,0,0},
  {3,7,0,0,0,0,0,0},{0,3,7,0,0,0,0,0},{1,3,7,0,0,0,0,0},{0,1,3,7,0,0,0,0},{2,3,7,0,0,0,0,0},{0,2,3,7,0,0,0,0},{1,2,3,7,0,0,0,0},{0,1,2,3,7,0,0,0},
  {4,7,0,0,0,0,0,0},{0,4,7,0,0,0,0,0},{1,4,7,0,0,0,0,0},{0,1,4,7,0,0,0,0},{2,4,7,0,0,0,0,0},{0,2,4,7,0,0,0,0},{1,2,4,7,0,0,0,0},{0,1,2,4,7,0,0,0},
  {3,4,7,0,0,0,0,0},{0,3,4,7,0,0,0,0},{1,3,4,7,0,0,0,0},{0,1,3,4,7,0,0,0},{2,3,4,7,0,0,0,0},{0,2,3,4,7,0,0,0},{1,2,3,4,7,0,0,0},{0,1,2,3,4,7,0,0},
  {5,7,0,0,0,0,0,0},{0,5,7,0,0,0,0,0},{1,5,7,0,0,0,0,0},{0,1,5,7,0,0,0,0},{2,5,7,0,0,0,0,0},{0,2,5,7,0,0,0,0},{1,2,5,7,0,0,0,0},{0,1,2,5,7,0,0,0},
  {3,5,7,0,0,0,0,0},{0,3,5,7,0,0,0,0},{1,3,5,7,0,0,0,0},{0,1,3,5,7,0,0,0},{2,3,5,7,0,0,0,0},{0,2,3,5,7,0,0,0},{1,2,3,5,7,0,0,0},{0,1,2,3,5,7,0,0},
  {4,5,7,0,0,0,0,0},{0,4,5,7,0,0,0,0},{1,4,5,7,0,0,0,0},{0,1,4,5,7,0,0,0},{2,4,5,7,0,0,0,0},{0,2,4,5,7,0,0,0},{1,2,4,5,7,0,0,0},{0,1,2,4,5,7,0,0},
  {3,4,5,7,0,0,0,0},{0,3,4,5,7,0,0,0},{1,3,4,5,7,0,0,0},{0,1,3,4,5,7,0,0},{2,3,4,5,7,0,0,0},{0,2,3,4,5,7,0,0},{1,2,3,4,5,7,0,0},{0,1,2,3,4,5,7,0},
  {6,7,0,0,0,0,0,0},{0,6,7,0,0,0,0,0},{1,6,7,0,0,0,0,0},{0,1,6,7,0,0,0,0},{2,6,7,0,0,0,0,0},{0,2,6,7,0,0,0,0},{1,2,6,7,0,0,0,0},{0,1,2,6,7,0,0,0},
  {3,6,7,0,0,0,0,0},{0,3,6,7,0,0,0,0},{1,3,6,7,0,0,0,0},{0,1,3,6,7,0,0,0},{2,3,6,7,0,0,0,0},{0,2,3,6,7,0,0,0},{1,2,3,6,7,0,0,0},{0,1,2,3,6,7,0,0},
  {4,6,7,0,0,0,0,0},{0,4,6,7,0,0,0,0},{1,4,6,7,0,0,0,0},{0,1,4,6,7,0,0,0},{2,4,6,7,0,0,0,0},{0,2,4,6,7,0,0,0},{1,2,4,6,7,0,0,0},{0,1,2,4,6,7,0,0},
  {3,4,6,7,0,0,0,0},{0,3,4,6,7,0,0,0},{1,3,4,6,7,0,0,0},{0,1,3,4,6,7,0,0},{2,3,4,6,7,0,0,0},{0,2,3,4,6,7,0,0},{1,2,3,4,6,7,0,0},{0,1,2,3,4,6,7,0},
  {5,6,7,0,0,0,0,0},{0,5,6,7,0,0,0,0},{1,5,6,7,0,0,0,0},{0,1,5,6,7,0,0,0},{2,5,6,7,0,0,0,0},{0,2,5,6,7,0,0,0},{1,2,5,6,7,0,0,0},{0,1,2,5,6,7,0,0},
  {3,5,6,7,0,0,0,0},{0,3,5,6,7,0,0,0},{1,3,5,6,7,0,0,0},{0,1,3,5,6,7,0,0},{2,3,5,6,7,0,0,0},{0,2,3,5,6,7,0,0},{1,2,3,5,6,7,0,0},{0,1,2,3,5,6,7,0},
  {4,5,6,7,0,0,0,0},{0,4,5,6,7,0,0,0},{1,4,5,6,7,0,0,0},{0,1,4,5,6,7,0,0},{2,4,5,6,7,0,0,0},{0,2,4,5,6,7,0,0},{1,2,4,5,6,7,0,0},{0,1,2,4,5,6,7,0},
  {3,4,5,6,7,0,0,0},{0,3,4,5,6,7,0,0},{1,3,4,5,6,7,0,0},{0,1,3,4,5,6,7,0},{2,3,4,5,6,7,0,0},{0,2,3,4,5,6,7,0},{1,2,3,4,5,6,7,0},{0,1,2,3,4,5,6,7}
};

int64_t segmented_bit_sieve_wheel_segment(int64_t k_low_start, int64_t k_low_stop, int64_t n_i_mod_bW , int64_t k_i, int64_t n_mod_bW, int64_t bW, std::vector<int64_t> RW, std::vector<int64_t> C1, std::vector<int64_t> C2, std::vector<uint8_t> Segment_0, int64_t p_mask_i, std::vector<uint8_t> Segment_mask, int ck_seg)
{
    //if ck_seg==1    returns the count of prime numbers in segments from  k_low_start to k_low_stop
    //if ck_seg==0    returns the count of prime numbers in first segment from  k_i to k_low_stop
    //if ck_seg==2    returns the count of prime numbers in last segments from  k_low_start to k_end
    int64_t  count_p = (int64_t)0;

    int64_t nR  = RW.size();
    int64_t nB = nR / 8; //nB number of byte for residue classes equal to nR=8*nB        
    int64_t segment_size_b = Segment_mask.size() - nB;
    int64_t segment_size = segment_size_b / nB;


    int64_t segment_size_0 = (Segment_0.size() / nB) - 1;        

    int64_t  p , pb , mb , mb_0 , ip , i , jb , j , jp , k , kmax , kb = 0;
    uint8_t  val_B;

    //vector used for segment of size (nB+segment_size_b)=nB*(1+segment_size)       
    std::vector<uint8_t> Segment_t(nB + segment_size_b);

    //segment scan
    int64_t k_low , kb_low;
    for (k_low = k_low_start; k_low < k_low_stop; k_low += segment_size)
    {
        kb_low = k_low * nB;
        //initialize segment    
        for (kb = (int64_t)0; kb < (nB + segment_size_b); kb++)
            Segment_t[kb] = Segment_mask[kb];
        //sieve for the segment
        kmax = (std::min(segment_size_0 , (int64_t) std::sqrt((k_low + segment_size) / bW) + 2));
        j = p_mask_i;
        for(k = (int64_t) 1; k <= kmax; k++)
        {
            kb = k * nB;
            mb_0 = -k_low + bW * k * k;    
            for (jb = 0; jb < nB; jb++)
            {
                val_B = Segment_0[kb +jb];
                for (; j < bit_count[val_B]; j++)
                {
                    jp = bit_pos[val_B][j] + jb * 8;
                    p = bW * k + RW[jp];
                    pb = p * nB;
                    for (ip = 0; ip < nR; ip++)
                    {
                            mb = mb_0 + k * C1[ip * nR + jp] + C2[ip * nR + jp];
                            if (mb < 0)
                                mb = (mb % p + p) % p;
                            if (mb <= segment_size)
                            {
                                mb *= nB;
                                mb += ip / 8;
                                i = ip % 8;
                                for (; mb < (nB + segment_size_b); mb += pb)
                                    Segment_t[mb] &= del_bit[i];
                            }
                    }
                }
                j = (int64_t) 0;
            }
        }
        //count prime numbers of the segment
        if (ck_seg == 1)
        {			
            for ( kb = nB + kb_low; kb < kb_low + segment_size_b + nB; kb++)
                count_p += bit_count[Segment_t[kb - kb_low]];
        }
        else if (ck_seg == 2)
        {			
            for ( kb = nB + kb_low; kb < std::min (kb_low + segment_size_b + nB , nB * k_low_stop); kb++)
                count_p += bit_count[Segment_t[kb - kb_low]];
        }
    }
    if (ck_seg == 0)
    {
        if (k_i > k_low_start && k_i <= k_low_start + segment_size)
        {
            //count prime numbers for k = k_i if k_i > segment_size_0
            if(n_i_mod_bW <= 1 && k_i > 1 + k_low_start)
            {    
                k_i--;
                if(Segment_t[(k_i - k_low_start) * nB + nB - 1] & (1 << 7))
                    count_p++;
            }
            else
            {    
                for (ip = 0; ip < nR; ip++)
                    if(Segment_t[(k_i - k_low_start) * nB + ip / 8] & (1 << ip % 8) && RW[ip] >= (n_i_mod_bW - bW))
                        count_p++;
            }
        }
        //count prime numbers of the segment
        if (k_i < k_low_start + segment_size)
        {
            for (kb = nB + k_i * nB; kb < std::min (nB * k_low_start + segment_size_b + nB , nB * k_low_stop); kb++)
                count_p += bit_count[Segment_t[kb - nB * k_low_start]];
        }
        if (k_low_start + segment_size ==  k_low_stop)
        {
            while(kb < nB * k_low_start + segment_size_b + nB)
            {
                count_p += bit_count[Segment_t[kb - nB * k_low_start]];
                kb++;
            }
        }
    }
    if (ck_seg != 1 && kb == nB * k_low_stop)
    {
        //count prime numbers for k=k_end=k_low_stop
        if ( kb - kb_low <= segment_size_b && kb - kb_low > (int64_t) 0)
            for (ip = 0; ip < nR; ip++)
                if(Segment_t[kb - kb_low + ip / 8]& (1 << ip % 8) && RW[ip] < (n_mod_bW - bW))
                    count_p++;
    }

    return count_p;
}

int64_t segmented_bit_sieve_wheel_IF_MT(uint64_t n_i, uint64_t n, int64_t max_bW, int64_t  max_threads)
{
    //returns the count of prime numbers >= n_i and < n
    //max_bW is max base wheel size choice max_bW = 30 , 210 , 2310
    //max_threads is maximum number of threads to use

    int64_t segment_size = 1; //initial segment size can be scaled up to have larger segments
    int64_t p_mask_i = 4;    //0 =< p_mask_i < 8 number primes following those of the basis for pre-sieve vector mask 
    
    int64_t sqrt_n = (int64_t) std::sqrt(n);

    int64_t  count_p = (int64_t)0;

    int64_t n_PB = (int64_t) 3;    
    int64_t bW = (int64_t) 30;
    
    //get bW base wheel equal to p1*p2*...*pn <=min(max_bW,sqrt_n)  with n=n_PB
    while(n_PB < n_PB_max && (bW * Primes_Base[n_PB] <= std::min(max_bW , sqrt_n)))
    {
        bW *= Primes_Base[n_PB];
        n_PB++;
    }

    for (int64_t i = 0; i < n_PB; i++)
        if (n > (uint64_t) Primes_Base[i] && (uint64_t) Primes_Base[i] >= n_i)
            count_p++;

    if (n > (uint64_t) (1 + Primes_Base[n_PB - 1]) && n > n_i)
    {

        int64_t k_i = (n_i < (uint64_t)2) ? 0 : (int64_t) (n_i / (uint64_t) bW + 1);
        int64_t k_end = (n < (uint64_t)bW) ? (int64_t) 2 : (int64_t) (n / (uint64_t) bW + 1);
        int64_t n_i_mod_bW = (n_i < (uint64_t)1) ? 0 : (int64_t) (n_i % (uint64_t) bW);
        int64_t n_mod_bW = (int64_t) (n % (uint64_t) bW);
        int64_t k_sqrt = (int64_t) std::sqrt(k_end / bW) + 1;

        //find reduct residue set modulo bW
        std::vector<char> Remainder_t(bW,true); 
        for (int64_t i = 0; i < n_PB; i++)
            for (int64_t j = Primes_Base[i]; j < bW; j += Primes_Base[i])
                Remainder_t[j] = false;
        std::vector<int64_t> RW;
        for (int64_t j = 2; j < bW; j++)
            if (Remainder_t[j] == true)
                RW.push_back(-bW + j);
        RW.push_back((int64_t)1);
        int64_t  nR = RW.size();   //nR=phi(bW)

        //get the matrix constant C1 and C2 to find the initial multiples of prime numbers in segment
        std::vector<int64_t> C1(nR * nR);
        std::vector<int64_t> C2(nR * nR);
        for (int64_t j = 0; j < nR - 2; j++)
        {
            int64_t rW_t , rW_t1;
            int64_t    j1=0;
            while((RW[j] * RW[j1]) % bW != bW - 1 && j1 < nR - 1)
                j1++;                
            rW_t1 = RW[j1];
            for (int64_t i = 0; i < nR; i++)
            {
                if (i == j)
                {
                    C2[nR * i + j] = 0;
                    C1[nR * i + j] = RW[j] + 1;
                }
                else if(i == nR - 3 - j)
                {
                    C2[nR * i + j] = 1;
                    C1[nR * i + j] = RW[j] - 1;
                }
                else
                {
                    rW_t = (int64_t) (rW_t1 * (-RW[i])) % bW;
                    if (rW_t > 1)
                        rW_t -= bW;
                    C1[nR * i + j] = rW_t + RW[j];
                    C2[nR * i + j] = (int64_t) (rW_t * RW[j]) / bW + 1;
                    if (i == nR - 1)
                        C2[nR * i + j] -= 1;
                }
            }
            C2[nR * j + nR - 2] = (int64_t) 1;
            C1[nR * j + nR - 2] = -(bW + RW[j]) - 1;
            C1[nR * j + nR - 1] = RW[j] + 1;
            C2[nR * j + nR - 1] = (int64_t )0;
        }
        for (int64_t i = nR - 2; i < nR; i++)
        {
            C2[nR * i + nR - 2] = (int64_t) 0;
            C1[nR * i + nR - 2] = -RW[i] - 1;
            C1[nR * i + nR - 1] = RW[i] + 1;
            C2[nR * i + nR - 1] = (int64_t) 0;
        }

        // get segment_size=p_(n_PB+1)*p_(n_PB+2)*...*p_(n_PB+p_mask_i)
        p_mask_i = std::min(p_mask_i, (int64_t)7); //p_mask_i < 8    
        p_mask_i = std::max(p_mask_i, (int64_t)0); //p_mask_i >=0
        if (p_mask_i < 2)
            segment_size = std::max(segment_size, (int64_t)128);            
        for (int64_t i = 0; i < p_mask_i; i++)
            segment_size *= (bW + RW[i]);
        int64_t segment_size_0 = segment_size;
        while (segment_size_0 < k_sqrt)
            segment_size_0 += segment_size;

        int64_t  nB = nR / 8; //nB number of byte for residue of congruence class equal to nR=8*nB        
        int64_t segment_size_b = nB * segment_size;
        
        //vector used for the first segment containing prime numbers less than sqrt(n)
        std::vector<uint8_t> Segment_0(nB + nB * segment_size_0, 0xff);
 
        int64_t  p , pb , mb , mb_0 , ip , i , jb , j , jp , k , kb = 0;
        uint8_t  val_B;
        int64_t kmax = (int64_t) std::sqrt(segment_size_0 / bW) + 1;
        //sieve for the first segment - includes prime numbers < sqrt(n)
        for (k = (int64_t)1; k  <= kmax; k++)
        {
            kb = nB * k;
            mb_0 = kb * k * bW;     //nB * k * k  * bW     
            for (jb = 0; jb < nB; jb++)
            {
                val_B = Segment_0[kb +jb];
                for (j = 0; j < bit_count[val_B]; j++)
                {
                    jp = bit_pos[val_B][j] + jb * 8;
                    pb = bW * kb + nB * RW[jp];  // nB * (bW * k + RW[jp]); 
                    for (ip = 0; ip < nR; ip++)
                    {
                            mb = ip / 8 + mb_0 + kb * C1[ip * nR + jp] + nB * C2[ip * nR + jp];
                            i = ip % 8;
                            for (; mb < (nB + nB * segment_size_0); mb += pb)
                                Segment_0[mb] &= del_bit[i];
                    }
                }
            }
        }

        if (k_i > 0 && k_i <= segment_size_0)
        {
            //count prime numbers for k = k_i
            if(n_i_mod_bW <= 1 && k_i > 1)
            {
                k_i--;
                if(Segment_0[k_i * nB + nB - 1] & (1 << 7))
                    count_p++;
            }
            else 
            {                
                for (ip = 0; ip < nR; ip++)
                    if(Segment_0[k_i * nB + ip / 8] & (1 << ip % 8) && RW[ip] >= (n_i_mod_bW - bW))
                        count_p++;
            }
        }
        if (k_i < segment_size_0)
        {
            //count the prime numbers in the first segment which contains k_i
            for (kb = nB * (k_i + 1); kb < std::min (nB + nB * segment_size_0 , nB * k_end); kb++)
                count_p += bit_count[Segment_0[kb]];
            //count prime numbers for k = k_end if k_end < segment_size_0
            if (kb == nB * k_end && kb <= nB * segment_size_0 && kb > 0)
                for (ip = 0; ip < nR; ip++)
                    if(Segment_0[kb + ip / 8]& (1 << ip % 8) && RW[ip] < (n_mod_bW - bW))
                        count_p++;
        }
        
        if (k_end > segment_size_0) 
        {
            // vector mask pre-sieve multiples of primes bW+RW[j]  with 0<j<p_mask_i
            std::vector<uint8_t> Segment_mask(nB + segment_size_b , 0xff);
            for (j = 0; j < p_mask_i; j++)
            {
                p = bW+RW[j];
                pb = p * nB;                
                for (ip = 0; ip < nR; ip++)
                {
                        mb = -segment_size_0 + bW + C1[ip * nR + j] + C2[ip * nR + j];
                        if (mb < 0)
                            mb=(mb % p + p) % p;
                        if (mb <= segment_size)
                        {
                            mb *= nB;
                            mb += ip / 8;
                            i = ip % 8;                            
                            for (; mb < (nB + segment_size_b); mb += pb)
                                Segment_mask[mb] &= del_bit[i];
                        }
                }
            }

            int64_t k_start = segment_size_0;
            if (k_i > segment_size_0)
            {
                //scanning the first segment and count from k_i
                k_start = segment_size * (k_i / segment_size);
                count_p += segmented_bit_sieve_wheel_segment(k_start, std::min (k_start + segment_size , k_end) ,n_i_mod_bW, k_i ,n_mod_bW , bW, RW, C1, C2, Segment_0, p_mask_i, Segment_mask , 0);
                k_start += segment_size;
            }

            int64_t range_thread = 0;
            int64_t n_threads = 1;
            //multithreaded section
            if (max_threads > (int64_t) 1)
            {
                n_threads = omp_get_max_threads();
                n_threads = std::min(n_threads, max_threads);

                while((k_end - k_start) / n_threads < segment_size && n_threads > 1)
                    n_threads --;
                if ((k_end - k_start) % n_threads && (k_end - k_start) % segment_size == 0 && n_threads > 1)
                {
                    if ((k_end - k_start) / n_threads == segment_size)
                        n_threads --;
                    else
                        range_thread = ((k_end - segment_size - k_start) / n_threads);
                }
                
                if (n_threads > 1)
                {
                    if (range_thread == 0)
                        range_thread = ((k_end - k_start) / n_threads);
                    range_thread = segment_size * (range_thread / segment_size);
  
                    std::vector<int64_t> count_p_vector(n_threads, 0);

                    #pragma omp parallel num_threads(n_threads)
                    {
                        int64_t t = omp_get_thread_num();
                        int64_t thread_start = k_start + t * range_thread;
                        int64_t thread_stop = thread_start + range_thread;
                        count_p_vector[t] = segmented_bit_sieve_wheel_segment(thread_start, thread_stop,n_i_mod_bW, k_i ,n_mod_bW , bW, RW, C1, C2, Segment_0, p_mask_i, Segment_mask , 1);
                    }
                    for (int64_t t = 0; t < n_threads; t++) 
                        count_p += count_p_vector[t];
                }
            }
            //end multithreaded section
            
            //scanning the last remaining segments and count to k_end          
            count_p += segmented_bit_sieve_wheel_segment(k_start + n_threads * range_thread, k_end ,n_i_mod_bW, k_i ,n_mod_bW , bW, RW, C1, C2, Segment_0, p_mask_i, Segment_mask , 2);
        }
    }

    return count_p;
}

int main()
{
    int64_t n_start = 0;
    int64_t n_stop = 1000000000;

    int64_t count = 0;

    //segmented_bit_sieve_wheel_IF_MT(n_start, n_stop, max_bW, max_threads) 
    //with max base wheel size choice max_bW= 30 , 210 , 2310 and max_threads maximum number of threads to use
    count = segmented_bit_sieve_wheel_IF_MT(n_start, n_stop, 210, 1);
 
    std::cout << " found " << count << " prime numbers >= " << n_start << " and < " << n_stop << std::endl; 

    return 0;
}
