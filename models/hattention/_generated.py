import sys
import torch
from typing import List, Tuple
from .base import make_levels_matrix

# hack
sys.setrecursionlimit(10000)


def level_lut_block_codegen(
    length: int,
    list_of_blocksize_target_source: List[Tuple[int, int, int]],
    debug: bool = False,
) -> None:

    if debug:
        maybe_comment = "# "
    else:
        maybe_comment = ""

    static_assert_bodies = []
    static_assert_headers = []
    for (lb, bt, bs) in list_of_blocksize_target_source:
        static_assert_bodies.append(f"(LB == {lb} and (BT == {bt} and BS == {bs}))")
        static_assert_headers.append("(")
    
    static_assert_body = "    " + "\n        ) or\n            ".join(static_assert_bodies)
    static_assert_header = "".join(static_assert_headers[1:])  # skip one bracket

    print(f"""
import triton
import triton.language as tl


{maybe_comment}@triton.jit
def level_lut_block(
    t,
    s,
    T: tl.constexpr,
    LB: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
):
    {maybe_comment}tl.static_assert(T == {length})
    {maybe_comment}tl.static_assert(
    {maybe_comment}    {static_assert_header}
    {maybe_comment}    {static_assert_body}
    {maybe_comment})
""")

    # this will cause out-of-bound indexing error
    # so we will know the conditions are not met
    print(f"""    v = -1""")
    for (LB, BT, BS) in list_of_blocksize_target_source:
        level_lut = make_levels_matrix(
            length=length,
            base=LB,
            dtype=torch.int64,
            device="cuda")

        print(f"    if (LB == {LB} and (BT == {BT} and BS == {BS})):")
        for t0 in range(0, length, BT):

            levels_to_print = []
            for s0 in range(0, length, BS):
                t1 = t0 + BT
                s1 = s0 + BS
                if t1 > length or s1 > length:
                    raise ValueError
                if s0 >= t0 or s1 >= t1:
                    if (len(levels_to_print) == 0 or
                        levels_to_print[-1]["level"] != -1):
                        levels_to_print.append({"index": s0, "level": -1})
                    continue

                levels = level_lut[t0: t1, s0: s1]
                level = torch.unique(levels).item()

                # adding a new level to print
                if (len(levels_to_print) == 0 or
                    levels_to_print[-1]["level"] != level):
                    levels_to_print.append({"index": s0, "level": level})

            if len(levels_to_print) == 0:
                raise ValueError

            if len(levels_to_print) == 1:
                if levels_to_print[0]["level"] != -1:
                    raise ValueError
                print(f"        {'if  ' if t0 == 0 else 'elif'}     ({t0:<5} <= t) and (t < {t1:<5}): v = -1")

            if len(levels_to_print) > 1:
                print(f"        {'if  ' if t0 == 0 else 'elif'}     ({t0:<5} <= t) and (t < {t1:<5}):")
                for i in range(len(levels_to_print) - 1):
                    level0 = levels_to_print[i]["level"]
                    index0 = levels_to_print[i]["index"]
                    index1 = levels_to_print[i + 1]["index"]

                    if i == 0:
                        print(f"            if   ({index0:<5} <= s) and (s < {index1:<5}): v = {level0}")
                    else:
                        print(f"            elif ({index0:<5} <= s) and (s < {index1:<5}): v = {level0}")

                print(f"            else                             : v = -1")

        print(f"        else                                 : v = -1")

    print(f"""    return v""")


import triton
import triton.language as tl


@triton.jit
def level_lut_block(
    t,
    s,
    T: tl.constexpr,
    LB: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
):
    tl.static_assert(T == 16384)
    tl.static_assert(
        ((
            (LB == 2 and (BT == 64 and BS == 64))
        ) or
            (LB == 2 and (BT == 128 and BS == 128))
        ) or
            (LB == 2 and (BT == 256 and BS == 256))
    )

    v = -1
    if (LB == 2 and (BT == 64 and BS == 64)):
        if       (0     <= t) and (t < 64   ): v = -1
        elif     (64    <= t) and (t < 128  ):
            if   (0     <= s) and (s < 64   ): v = 7
            else                             : v = -1
        elif     (128   <= t) and (t < 192  ):
            if   (0     <= s) and (s < 128  ): v = 8
            else                             : v = -1
        elif     (192   <= t) and (t < 256  ):
            if   (0     <= s) and (s < 128  ): v = 8
            elif (128   <= s) and (s < 192  ): v = 7
            else                             : v = -1
        elif     (256   <= t) and (t < 320  ):
            if   (0     <= s) and (s < 256  ): v = 9
            else                             : v = -1
        elif     (320   <= t) and (t < 384  ):
            if   (0     <= s) and (s < 256  ): v = 9
            elif (256   <= s) and (s < 320  ): v = 7
            else                             : v = -1
        elif     (384   <= t) and (t < 448  ):
            if   (0     <= s) and (s < 256  ): v = 9
            elif (256   <= s) and (s < 384  ): v = 8
            else                             : v = -1
        elif     (448   <= t) and (t < 512  ):
            if   (0     <= s) and (s < 256  ): v = 9
            elif (256   <= s) and (s < 384  ): v = 8
            elif (384   <= s) and (s < 448  ): v = 7
            else                             : v = -1
        elif     (512   <= t) and (t < 576  ):
            if   (0     <= s) and (s < 512  ): v = 10
            else                             : v = -1
        elif     (576   <= t) and (t < 640  ):
            if   (0     <= s) and (s < 512  ): v = 10
            elif (512   <= s) and (s < 576  ): v = 7
            else                             : v = -1
        elif     (640   <= t) and (t < 704  ):
            if   (0     <= s) and (s < 512  ): v = 10
            elif (512   <= s) and (s < 640  ): v = 8
            else                             : v = -1
        elif     (704   <= t) and (t < 768  ):
            if   (0     <= s) and (s < 512  ): v = 10
            elif (512   <= s) and (s < 640  ): v = 8
            elif (640   <= s) and (s < 704  ): v = 7
            else                             : v = -1
        elif     (768   <= t) and (t < 832  ):
            if   (0     <= s) and (s < 512  ): v = 10
            elif (512   <= s) and (s < 768  ): v = 9
            else                             : v = -1
        elif     (832   <= t) and (t < 896  ):
            if   (0     <= s) and (s < 512  ): v = 10
            elif (512   <= s) and (s < 768  ): v = 9
            elif (768   <= s) and (s < 832  ): v = 7
            else                             : v = -1
        elif     (896   <= t) and (t < 960  ):
            if   (0     <= s) and (s < 512  ): v = 10
            elif (512   <= s) and (s < 768  ): v = 9
            elif (768   <= s) and (s < 896  ): v = 8
            else                             : v = -1
        elif     (960   <= t) and (t < 1024 ):
            if   (0     <= s) and (s < 512  ): v = 10
            elif (512   <= s) and (s < 768  ): v = 9
            elif (768   <= s) and (s < 896  ): v = 8
            elif (896   <= s) and (s < 960  ): v = 7
            else                             : v = -1
        elif     (1024  <= t) and (t < 1088 ):
            if   (0     <= s) and (s < 1024 ): v = 11
            else                             : v = -1
        elif     (1088  <= t) and (t < 1152 ):
            if   (0     <= s) and (s < 1024 ): v = 11
            elif (1024  <= s) and (s < 1088 ): v = 7
            else                             : v = -1
        elif     (1152  <= t) and (t < 1216 ):
            if   (0     <= s) and (s < 1024 ): v = 11
            elif (1024  <= s) and (s < 1152 ): v = 8
            else                             : v = -1
        elif     (1216  <= t) and (t < 1280 ):
            if   (0     <= s) and (s < 1024 ): v = 11
            elif (1024  <= s) and (s < 1152 ): v = 8
            elif (1152  <= s) and (s < 1216 ): v = 7
            else                             : v = -1
        elif     (1280  <= t) and (t < 1344 ):
            if   (0     <= s) and (s < 1024 ): v = 11
            elif (1024  <= s) and (s < 1280 ): v = 9
            else                             : v = -1
        elif     (1344  <= t) and (t < 1408 ):
            if   (0     <= s) and (s < 1024 ): v = 11
            elif (1024  <= s) and (s < 1280 ): v = 9
            elif (1280  <= s) and (s < 1344 ): v = 7
            else                             : v = -1
        elif     (1408  <= t) and (t < 1472 ):
            if   (0     <= s) and (s < 1024 ): v = 11
            elif (1024  <= s) and (s < 1280 ): v = 9
            elif (1280  <= s) and (s < 1408 ): v = 8
            else                             : v = -1
        elif     (1472  <= t) and (t < 1536 ):
            if   (0     <= s) and (s < 1024 ): v = 11
            elif (1024  <= s) and (s < 1280 ): v = 9
            elif (1280  <= s) and (s < 1408 ): v = 8
            elif (1408  <= s) and (s < 1472 ): v = 7
            else                             : v = -1
        elif     (1536  <= t) and (t < 1600 ):
            if   (0     <= s) and (s < 1024 ): v = 11
            elif (1024  <= s) and (s < 1536 ): v = 10
            else                             : v = -1
        elif     (1600  <= t) and (t < 1664 ):
            if   (0     <= s) and (s < 1024 ): v = 11
            elif (1024  <= s) and (s < 1536 ): v = 10
            elif (1536  <= s) and (s < 1600 ): v = 7
            else                             : v = -1
        elif     (1664  <= t) and (t < 1728 ):
            if   (0     <= s) and (s < 1024 ): v = 11
            elif (1024  <= s) and (s < 1536 ): v = 10
            elif (1536  <= s) and (s < 1664 ): v = 8
            else                             : v = -1
        elif     (1728  <= t) and (t < 1792 ):
            if   (0     <= s) and (s < 1024 ): v = 11
            elif (1024  <= s) and (s < 1536 ): v = 10
            elif (1536  <= s) and (s < 1664 ): v = 8
            elif (1664  <= s) and (s < 1728 ): v = 7
            else                             : v = -1
        elif     (1792  <= t) and (t < 1856 ):
            if   (0     <= s) and (s < 1024 ): v = 11
            elif (1024  <= s) and (s < 1536 ): v = 10
            elif (1536  <= s) and (s < 1792 ): v = 9
            else                             : v = -1
        elif     (1856  <= t) and (t < 1920 ):
            if   (0     <= s) and (s < 1024 ): v = 11
            elif (1024  <= s) and (s < 1536 ): v = 10
            elif (1536  <= s) and (s < 1792 ): v = 9
            elif (1792  <= s) and (s < 1856 ): v = 7
            else                             : v = -1
        elif     (1920  <= t) and (t < 1984 ):
            if   (0     <= s) and (s < 1024 ): v = 11
            elif (1024  <= s) and (s < 1536 ): v = 10
            elif (1536  <= s) and (s < 1792 ): v = 9
            elif (1792  <= s) and (s < 1920 ): v = 8
            else                             : v = -1
        elif     (1984  <= t) and (t < 2048 ):
            if   (0     <= s) and (s < 1024 ): v = 11
            elif (1024  <= s) and (s < 1536 ): v = 10
            elif (1536  <= s) and (s < 1792 ): v = 9
            elif (1792  <= s) and (s < 1920 ): v = 8
            elif (1920  <= s) and (s < 1984 ): v = 7
            else                             : v = -1
        elif     (2048  <= t) and (t < 2112 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            else                             : v = -1
        elif     (2112  <= t) and (t < 2176 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 2112 ): v = 7
            else                             : v = -1
        elif     (2176  <= t) and (t < 2240 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 2176 ): v = 8
            else                             : v = -1
        elif     (2240  <= t) and (t < 2304 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 2176 ): v = 8
            elif (2176  <= s) and (s < 2240 ): v = 7
            else                             : v = -1
        elif     (2304  <= t) and (t < 2368 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 2304 ): v = 9
            else                             : v = -1
        elif     (2368  <= t) and (t < 2432 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 2304 ): v = 9
            elif (2304  <= s) and (s < 2368 ): v = 7
            else                             : v = -1
        elif     (2432  <= t) and (t < 2496 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 2304 ): v = 9
            elif (2304  <= s) and (s < 2432 ): v = 8
            else                             : v = -1
        elif     (2496  <= t) and (t < 2560 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 2304 ): v = 9
            elif (2304  <= s) and (s < 2432 ): v = 8
            elif (2432  <= s) and (s < 2496 ): v = 7
            else                             : v = -1
        elif     (2560  <= t) and (t < 2624 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 2560 ): v = 10
            else                             : v = -1
        elif     (2624  <= t) and (t < 2688 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 2560 ): v = 10
            elif (2560  <= s) and (s < 2624 ): v = 7
            else                             : v = -1
        elif     (2688  <= t) and (t < 2752 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 2560 ): v = 10
            elif (2560  <= s) and (s < 2688 ): v = 8
            else                             : v = -1
        elif     (2752  <= t) and (t < 2816 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 2560 ): v = 10
            elif (2560  <= s) and (s < 2688 ): v = 8
            elif (2688  <= s) and (s < 2752 ): v = 7
            else                             : v = -1
        elif     (2816  <= t) and (t < 2880 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 2560 ): v = 10
            elif (2560  <= s) and (s < 2816 ): v = 9
            else                             : v = -1
        elif     (2880  <= t) and (t < 2944 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 2560 ): v = 10
            elif (2560  <= s) and (s < 2816 ): v = 9
            elif (2816  <= s) and (s < 2880 ): v = 7
            else                             : v = -1
        elif     (2944  <= t) and (t < 3008 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 2560 ): v = 10
            elif (2560  <= s) and (s < 2816 ): v = 9
            elif (2816  <= s) and (s < 2944 ): v = 8
            else                             : v = -1
        elif     (3008  <= t) and (t < 3072 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 2560 ): v = 10
            elif (2560  <= s) and (s < 2816 ): v = 9
            elif (2816  <= s) and (s < 2944 ): v = 8
            elif (2944  <= s) and (s < 3008 ): v = 7
            else                             : v = -1
        elif     (3072  <= t) and (t < 3136 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 3072 ): v = 11
            else                             : v = -1
        elif     (3136  <= t) and (t < 3200 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 3072 ): v = 11
            elif (3072  <= s) and (s < 3136 ): v = 7
            else                             : v = -1
        elif     (3200  <= t) and (t < 3264 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 3072 ): v = 11
            elif (3072  <= s) and (s < 3200 ): v = 8
            else                             : v = -1
        elif     (3264  <= t) and (t < 3328 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 3072 ): v = 11
            elif (3072  <= s) and (s < 3200 ): v = 8
            elif (3200  <= s) and (s < 3264 ): v = 7
            else                             : v = -1
        elif     (3328  <= t) and (t < 3392 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 3072 ): v = 11
            elif (3072  <= s) and (s < 3328 ): v = 9
            else                             : v = -1
        elif     (3392  <= t) and (t < 3456 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 3072 ): v = 11
            elif (3072  <= s) and (s < 3328 ): v = 9
            elif (3328  <= s) and (s < 3392 ): v = 7
            else                             : v = -1
        elif     (3456  <= t) and (t < 3520 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 3072 ): v = 11
            elif (3072  <= s) and (s < 3328 ): v = 9
            elif (3328  <= s) and (s < 3456 ): v = 8
            else                             : v = -1
        elif     (3520  <= t) and (t < 3584 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 3072 ): v = 11
            elif (3072  <= s) and (s < 3328 ): v = 9
            elif (3328  <= s) and (s < 3456 ): v = 8
            elif (3456  <= s) and (s < 3520 ): v = 7
            else                             : v = -1
        elif     (3584  <= t) and (t < 3648 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 3072 ): v = 11
            elif (3072  <= s) and (s < 3584 ): v = 10
            else                             : v = -1
        elif     (3648  <= t) and (t < 3712 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 3072 ): v = 11
            elif (3072  <= s) and (s < 3584 ): v = 10
            elif (3584  <= s) and (s < 3648 ): v = 7
            else                             : v = -1
        elif     (3712  <= t) and (t < 3776 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 3072 ): v = 11
            elif (3072  <= s) and (s < 3584 ): v = 10
            elif (3584  <= s) and (s < 3712 ): v = 8
            else                             : v = -1
        elif     (3776  <= t) and (t < 3840 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 3072 ): v = 11
            elif (3072  <= s) and (s < 3584 ): v = 10
            elif (3584  <= s) and (s < 3712 ): v = 8
            elif (3712  <= s) and (s < 3776 ): v = 7
            else                             : v = -1
        elif     (3840  <= t) and (t < 3904 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 3072 ): v = 11
            elif (3072  <= s) and (s < 3584 ): v = 10
            elif (3584  <= s) and (s < 3840 ): v = 9
            else                             : v = -1
        elif     (3904  <= t) and (t < 3968 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 3072 ): v = 11
            elif (3072  <= s) and (s < 3584 ): v = 10
            elif (3584  <= s) and (s < 3840 ): v = 9
            elif (3840  <= s) and (s < 3904 ): v = 7
            else                             : v = -1
        elif     (3968  <= t) and (t < 4032 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 3072 ): v = 11
            elif (3072  <= s) and (s < 3584 ): v = 10
            elif (3584  <= s) and (s < 3840 ): v = 9
            elif (3840  <= s) and (s < 3968 ): v = 8
            else                             : v = -1
        elif     (4032  <= t) and (t < 4096 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 3072 ): v = 11
            elif (3072  <= s) and (s < 3584 ): v = 10
            elif (3584  <= s) and (s < 3840 ): v = 9
            elif (3840  <= s) and (s < 3968 ): v = 8
            elif (3968  <= s) and (s < 4032 ): v = 7
            else                             : v = -1
        elif     (4096  <= t) and (t < 4160 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            else                             : v = -1
        elif     (4160  <= t) and (t < 4224 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 4160 ): v = 7
            else                             : v = -1
        elif     (4224  <= t) and (t < 4288 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 4224 ): v = 8
            else                             : v = -1
        elif     (4288  <= t) and (t < 4352 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 4224 ): v = 8
            elif (4224  <= s) and (s < 4288 ): v = 7
            else                             : v = -1
        elif     (4352  <= t) and (t < 4416 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 4352 ): v = 9
            else                             : v = -1
        elif     (4416  <= t) and (t < 4480 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 4352 ): v = 9
            elif (4352  <= s) and (s < 4416 ): v = 7
            else                             : v = -1
        elif     (4480  <= t) and (t < 4544 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 4352 ): v = 9
            elif (4352  <= s) and (s < 4480 ): v = 8
            else                             : v = -1
        elif     (4544  <= t) and (t < 4608 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 4352 ): v = 9
            elif (4352  <= s) and (s < 4480 ): v = 8
            elif (4480  <= s) and (s < 4544 ): v = 7
            else                             : v = -1
        elif     (4608  <= t) and (t < 4672 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 4608 ): v = 10
            else                             : v = -1
        elif     (4672  <= t) and (t < 4736 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 4608 ): v = 10
            elif (4608  <= s) and (s < 4672 ): v = 7
            else                             : v = -1
        elif     (4736  <= t) and (t < 4800 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 4608 ): v = 10
            elif (4608  <= s) and (s < 4736 ): v = 8
            else                             : v = -1
        elif     (4800  <= t) and (t < 4864 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 4608 ): v = 10
            elif (4608  <= s) and (s < 4736 ): v = 8
            elif (4736  <= s) and (s < 4800 ): v = 7
            else                             : v = -1
        elif     (4864  <= t) and (t < 4928 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 4608 ): v = 10
            elif (4608  <= s) and (s < 4864 ): v = 9
            else                             : v = -1
        elif     (4928  <= t) and (t < 4992 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 4608 ): v = 10
            elif (4608  <= s) and (s < 4864 ): v = 9
            elif (4864  <= s) and (s < 4928 ): v = 7
            else                             : v = -1
        elif     (4992  <= t) and (t < 5056 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 4608 ): v = 10
            elif (4608  <= s) and (s < 4864 ): v = 9
            elif (4864  <= s) and (s < 4992 ): v = 8
            else                             : v = -1
        elif     (5056  <= t) and (t < 5120 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 4608 ): v = 10
            elif (4608  <= s) and (s < 4864 ): v = 9
            elif (4864  <= s) and (s < 4992 ): v = 8
            elif (4992  <= s) and (s < 5056 ): v = 7
            else                             : v = -1
        elif     (5120  <= t) and (t < 5184 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 5120 ): v = 11
            else                             : v = -1
        elif     (5184  <= t) and (t < 5248 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 5120 ): v = 11
            elif (5120  <= s) and (s < 5184 ): v = 7
            else                             : v = -1
        elif     (5248  <= t) and (t < 5312 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 5120 ): v = 11
            elif (5120  <= s) and (s < 5248 ): v = 8
            else                             : v = -1
        elif     (5312  <= t) and (t < 5376 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 5120 ): v = 11
            elif (5120  <= s) and (s < 5248 ): v = 8
            elif (5248  <= s) and (s < 5312 ): v = 7
            else                             : v = -1
        elif     (5376  <= t) and (t < 5440 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 5120 ): v = 11
            elif (5120  <= s) and (s < 5376 ): v = 9
            else                             : v = -1
        elif     (5440  <= t) and (t < 5504 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 5120 ): v = 11
            elif (5120  <= s) and (s < 5376 ): v = 9
            elif (5376  <= s) and (s < 5440 ): v = 7
            else                             : v = -1
        elif     (5504  <= t) and (t < 5568 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 5120 ): v = 11
            elif (5120  <= s) and (s < 5376 ): v = 9
            elif (5376  <= s) and (s < 5504 ): v = 8
            else                             : v = -1
        elif     (5568  <= t) and (t < 5632 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 5120 ): v = 11
            elif (5120  <= s) and (s < 5376 ): v = 9
            elif (5376  <= s) and (s < 5504 ): v = 8
            elif (5504  <= s) and (s < 5568 ): v = 7
            else                             : v = -1
        elif     (5632  <= t) and (t < 5696 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 5120 ): v = 11
            elif (5120  <= s) and (s < 5632 ): v = 10
            else                             : v = -1
        elif     (5696  <= t) and (t < 5760 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 5120 ): v = 11
            elif (5120  <= s) and (s < 5632 ): v = 10
            elif (5632  <= s) and (s < 5696 ): v = 7
            else                             : v = -1
        elif     (5760  <= t) and (t < 5824 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 5120 ): v = 11
            elif (5120  <= s) and (s < 5632 ): v = 10
            elif (5632  <= s) and (s < 5760 ): v = 8
            else                             : v = -1
        elif     (5824  <= t) and (t < 5888 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 5120 ): v = 11
            elif (5120  <= s) and (s < 5632 ): v = 10
            elif (5632  <= s) and (s < 5760 ): v = 8
            elif (5760  <= s) and (s < 5824 ): v = 7
            else                             : v = -1
        elif     (5888  <= t) and (t < 5952 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 5120 ): v = 11
            elif (5120  <= s) and (s < 5632 ): v = 10
            elif (5632  <= s) and (s < 5888 ): v = 9
            else                             : v = -1
        elif     (5952  <= t) and (t < 6016 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 5120 ): v = 11
            elif (5120  <= s) and (s < 5632 ): v = 10
            elif (5632  <= s) and (s < 5888 ): v = 9
            elif (5888  <= s) and (s < 5952 ): v = 7
            else                             : v = -1
        elif     (6016  <= t) and (t < 6080 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 5120 ): v = 11
            elif (5120  <= s) and (s < 5632 ): v = 10
            elif (5632  <= s) and (s < 5888 ): v = 9
            elif (5888  <= s) and (s < 6016 ): v = 8
            else                             : v = -1
        elif     (6080  <= t) and (t < 6144 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 5120 ): v = 11
            elif (5120  <= s) and (s < 5632 ): v = 10
            elif (5632  <= s) and (s < 5888 ): v = 9
            elif (5888  <= s) and (s < 6016 ): v = 8
            elif (6016  <= s) and (s < 6080 ): v = 7
            else                             : v = -1
        elif     (6144  <= t) and (t < 6208 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            else                             : v = -1
        elif     (6208  <= t) and (t < 6272 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 6208 ): v = 7
            else                             : v = -1
        elif     (6272  <= t) and (t < 6336 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 6272 ): v = 8
            else                             : v = -1
        elif     (6336  <= t) and (t < 6400 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 6272 ): v = 8
            elif (6272  <= s) and (s < 6336 ): v = 7
            else                             : v = -1
        elif     (6400  <= t) and (t < 6464 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 6400 ): v = 9
            else                             : v = -1
        elif     (6464  <= t) and (t < 6528 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 6400 ): v = 9
            elif (6400  <= s) and (s < 6464 ): v = 7
            else                             : v = -1
        elif     (6528  <= t) and (t < 6592 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 6400 ): v = 9
            elif (6400  <= s) and (s < 6528 ): v = 8
            else                             : v = -1
        elif     (6592  <= t) and (t < 6656 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 6400 ): v = 9
            elif (6400  <= s) and (s < 6528 ): v = 8
            elif (6528  <= s) and (s < 6592 ): v = 7
            else                             : v = -1
        elif     (6656  <= t) and (t < 6720 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 6656 ): v = 10
            else                             : v = -1
        elif     (6720  <= t) and (t < 6784 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 6656 ): v = 10
            elif (6656  <= s) and (s < 6720 ): v = 7
            else                             : v = -1
        elif     (6784  <= t) and (t < 6848 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 6656 ): v = 10
            elif (6656  <= s) and (s < 6784 ): v = 8
            else                             : v = -1
        elif     (6848  <= t) and (t < 6912 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 6656 ): v = 10
            elif (6656  <= s) and (s < 6784 ): v = 8
            elif (6784  <= s) and (s < 6848 ): v = 7
            else                             : v = -1
        elif     (6912  <= t) and (t < 6976 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 6656 ): v = 10
            elif (6656  <= s) and (s < 6912 ): v = 9
            else                             : v = -1
        elif     (6976  <= t) and (t < 7040 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 6656 ): v = 10
            elif (6656  <= s) and (s < 6912 ): v = 9
            elif (6912  <= s) and (s < 6976 ): v = 7
            else                             : v = -1
        elif     (7040  <= t) and (t < 7104 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 6656 ): v = 10
            elif (6656  <= s) and (s < 6912 ): v = 9
            elif (6912  <= s) and (s < 7040 ): v = 8
            else                             : v = -1
        elif     (7104  <= t) and (t < 7168 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 6656 ): v = 10
            elif (6656  <= s) and (s < 6912 ): v = 9
            elif (6912  <= s) and (s < 7040 ): v = 8
            elif (7040  <= s) and (s < 7104 ): v = 7
            else                             : v = -1
        elif     (7168  <= t) and (t < 7232 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 7168 ): v = 11
            else                             : v = -1
        elif     (7232  <= t) and (t < 7296 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 7168 ): v = 11
            elif (7168  <= s) and (s < 7232 ): v = 7
            else                             : v = -1
        elif     (7296  <= t) and (t < 7360 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 7168 ): v = 11
            elif (7168  <= s) and (s < 7296 ): v = 8
            else                             : v = -1
        elif     (7360  <= t) and (t < 7424 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 7168 ): v = 11
            elif (7168  <= s) and (s < 7296 ): v = 8
            elif (7296  <= s) and (s < 7360 ): v = 7
            else                             : v = -1
        elif     (7424  <= t) and (t < 7488 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 7168 ): v = 11
            elif (7168  <= s) and (s < 7424 ): v = 9
            else                             : v = -1
        elif     (7488  <= t) and (t < 7552 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 7168 ): v = 11
            elif (7168  <= s) and (s < 7424 ): v = 9
            elif (7424  <= s) and (s < 7488 ): v = 7
            else                             : v = -1
        elif     (7552  <= t) and (t < 7616 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 7168 ): v = 11
            elif (7168  <= s) and (s < 7424 ): v = 9
            elif (7424  <= s) and (s < 7552 ): v = 8
            else                             : v = -1
        elif     (7616  <= t) and (t < 7680 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 7168 ): v = 11
            elif (7168  <= s) and (s < 7424 ): v = 9
            elif (7424  <= s) and (s < 7552 ): v = 8
            elif (7552  <= s) and (s < 7616 ): v = 7
            else                             : v = -1
        elif     (7680  <= t) and (t < 7744 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 7168 ): v = 11
            elif (7168  <= s) and (s < 7680 ): v = 10
            else                             : v = -1
        elif     (7744  <= t) and (t < 7808 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 7168 ): v = 11
            elif (7168  <= s) and (s < 7680 ): v = 10
            elif (7680  <= s) and (s < 7744 ): v = 7
            else                             : v = -1
        elif     (7808  <= t) and (t < 7872 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 7168 ): v = 11
            elif (7168  <= s) and (s < 7680 ): v = 10
            elif (7680  <= s) and (s < 7808 ): v = 8
            else                             : v = -1
        elif     (7872  <= t) and (t < 7936 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 7168 ): v = 11
            elif (7168  <= s) and (s < 7680 ): v = 10
            elif (7680  <= s) and (s < 7808 ): v = 8
            elif (7808  <= s) and (s < 7872 ): v = 7
            else                             : v = -1
        elif     (7936  <= t) and (t < 8000 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 7168 ): v = 11
            elif (7168  <= s) and (s < 7680 ): v = 10
            elif (7680  <= s) and (s < 7936 ): v = 9
            else                             : v = -1
        elif     (8000  <= t) and (t < 8064 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 7168 ): v = 11
            elif (7168  <= s) and (s < 7680 ): v = 10
            elif (7680  <= s) and (s < 7936 ): v = 9
            elif (7936  <= s) and (s < 8000 ): v = 7
            else                             : v = -1
        elif     (8064  <= t) and (t < 8128 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 7168 ): v = 11
            elif (7168  <= s) and (s < 7680 ): v = 10
            elif (7680  <= s) and (s < 7936 ): v = 9
            elif (7936  <= s) and (s < 8064 ): v = 8
            else                             : v = -1
        elif     (8128  <= t) and (t < 8192 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 7168 ): v = 11
            elif (7168  <= s) and (s < 7680 ): v = 10
            elif (7680  <= s) and (s < 7936 ): v = 9
            elif (7936  <= s) and (s < 8064 ): v = 8
            elif (8064  <= s) and (s < 8128 ): v = 7
            else                             : v = -1
        elif     (8192  <= t) and (t < 8256 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            else                             : v = -1
        elif     (8256  <= t) and (t < 8320 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 8256 ): v = 7
            else                             : v = -1
        elif     (8320  <= t) and (t < 8384 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 8320 ): v = 8
            else                             : v = -1
        elif     (8384  <= t) and (t < 8448 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 8320 ): v = 8
            elif (8320  <= s) and (s < 8384 ): v = 7
            else                             : v = -1
        elif     (8448  <= t) and (t < 8512 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 8448 ): v = 9
            else                             : v = -1
        elif     (8512  <= t) and (t < 8576 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 8448 ): v = 9
            elif (8448  <= s) and (s < 8512 ): v = 7
            else                             : v = -1
        elif     (8576  <= t) and (t < 8640 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 8448 ): v = 9
            elif (8448  <= s) and (s < 8576 ): v = 8
            else                             : v = -1
        elif     (8640  <= t) and (t < 8704 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 8448 ): v = 9
            elif (8448  <= s) and (s < 8576 ): v = 8
            elif (8576  <= s) and (s < 8640 ): v = 7
            else                             : v = -1
        elif     (8704  <= t) and (t < 8768 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 8704 ): v = 10
            else                             : v = -1
        elif     (8768  <= t) and (t < 8832 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 8704 ): v = 10
            elif (8704  <= s) and (s < 8768 ): v = 7
            else                             : v = -1
        elif     (8832  <= t) and (t < 8896 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 8704 ): v = 10
            elif (8704  <= s) and (s < 8832 ): v = 8
            else                             : v = -1
        elif     (8896  <= t) and (t < 8960 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 8704 ): v = 10
            elif (8704  <= s) and (s < 8832 ): v = 8
            elif (8832  <= s) and (s < 8896 ): v = 7
            else                             : v = -1
        elif     (8960  <= t) and (t < 9024 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 8704 ): v = 10
            elif (8704  <= s) and (s < 8960 ): v = 9
            else                             : v = -1
        elif     (9024  <= t) and (t < 9088 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 8704 ): v = 10
            elif (8704  <= s) and (s < 8960 ): v = 9
            elif (8960  <= s) and (s < 9024 ): v = 7
            else                             : v = -1
        elif     (9088  <= t) and (t < 9152 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 8704 ): v = 10
            elif (8704  <= s) and (s < 8960 ): v = 9
            elif (8960  <= s) and (s < 9088 ): v = 8
            else                             : v = -1
        elif     (9152  <= t) and (t < 9216 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 8704 ): v = 10
            elif (8704  <= s) and (s < 8960 ): v = 9
            elif (8960  <= s) and (s < 9088 ): v = 8
            elif (9088  <= s) and (s < 9152 ): v = 7
            else                             : v = -1
        elif     (9216  <= t) and (t < 9280 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 9216 ): v = 11
            else                             : v = -1
        elif     (9280  <= t) and (t < 9344 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 9216 ): v = 11
            elif (9216  <= s) and (s < 9280 ): v = 7
            else                             : v = -1
        elif     (9344  <= t) and (t < 9408 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 9216 ): v = 11
            elif (9216  <= s) and (s < 9344 ): v = 8
            else                             : v = -1
        elif     (9408  <= t) and (t < 9472 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 9216 ): v = 11
            elif (9216  <= s) and (s < 9344 ): v = 8
            elif (9344  <= s) and (s < 9408 ): v = 7
            else                             : v = -1
        elif     (9472  <= t) and (t < 9536 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 9216 ): v = 11
            elif (9216  <= s) and (s < 9472 ): v = 9
            else                             : v = -1
        elif     (9536  <= t) and (t < 9600 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 9216 ): v = 11
            elif (9216  <= s) and (s < 9472 ): v = 9
            elif (9472  <= s) and (s < 9536 ): v = 7
            else                             : v = -1
        elif     (9600  <= t) and (t < 9664 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 9216 ): v = 11
            elif (9216  <= s) and (s < 9472 ): v = 9
            elif (9472  <= s) and (s < 9600 ): v = 8
            else                             : v = -1
        elif     (9664  <= t) and (t < 9728 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 9216 ): v = 11
            elif (9216  <= s) and (s < 9472 ): v = 9
            elif (9472  <= s) and (s < 9600 ): v = 8
            elif (9600  <= s) and (s < 9664 ): v = 7
            else                             : v = -1
        elif     (9728  <= t) and (t < 9792 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 9216 ): v = 11
            elif (9216  <= s) and (s < 9728 ): v = 10
            else                             : v = -1
        elif     (9792  <= t) and (t < 9856 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 9216 ): v = 11
            elif (9216  <= s) and (s < 9728 ): v = 10
            elif (9728  <= s) and (s < 9792 ): v = 7
            else                             : v = -1
        elif     (9856  <= t) and (t < 9920 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 9216 ): v = 11
            elif (9216  <= s) and (s < 9728 ): v = 10
            elif (9728  <= s) and (s < 9856 ): v = 8
            else                             : v = -1
        elif     (9920  <= t) and (t < 9984 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 9216 ): v = 11
            elif (9216  <= s) and (s < 9728 ): v = 10
            elif (9728  <= s) and (s < 9856 ): v = 8
            elif (9856  <= s) and (s < 9920 ): v = 7
            else                             : v = -1
        elif     (9984  <= t) and (t < 10048):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 9216 ): v = 11
            elif (9216  <= s) and (s < 9728 ): v = 10
            elif (9728  <= s) and (s < 9984 ): v = 9
            else                             : v = -1
        elif     (10048 <= t) and (t < 10112):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 9216 ): v = 11
            elif (9216  <= s) and (s < 9728 ): v = 10
            elif (9728  <= s) and (s < 9984 ): v = 9
            elif (9984  <= s) and (s < 10048): v = 7
            else                             : v = -1
        elif     (10112 <= t) and (t < 10176):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 9216 ): v = 11
            elif (9216  <= s) and (s < 9728 ): v = 10
            elif (9728  <= s) and (s < 9984 ): v = 9
            elif (9984  <= s) and (s < 10112): v = 8
            else                             : v = -1
        elif     (10176 <= t) and (t < 10240):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 9216 ): v = 11
            elif (9216  <= s) and (s < 9728 ): v = 10
            elif (9728  <= s) and (s < 9984 ): v = 9
            elif (9984  <= s) and (s < 10112): v = 8
            elif (10112 <= s) and (s < 10176): v = 7
            else                             : v = -1
        elif     (10240 <= t) and (t < 10304):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            else                             : v = -1
        elif     (10304 <= t) and (t < 10368):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 10304): v = 7
            else                             : v = -1
        elif     (10368 <= t) and (t < 10432):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 10368): v = 8
            else                             : v = -1
        elif     (10432 <= t) and (t < 10496):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 10368): v = 8
            elif (10368 <= s) and (s < 10432): v = 7
            else                             : v = -1
        elif     (10496 <= t) and (t < 10560):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 10496): v = 9
            else                             : v = -1
        elif     (10560 <= t) and (t < 10624):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 10496): v = 9
            elif (10496 <= s) and (s < 10560): v = 7
            else                             : v = -1
        elif     (10624 <= t) and (t < 10688):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 10496): v = 9
            elif (10496 <= s) and (s < 10624): v = 8
            else                             : v = -1
        elif     (10688 <= t) and (t < 10752):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 10496): v = 9
            elif (10496 <= s) and (s < 10624): v = 8
            elif (10624 <= s) and (s < 10688): v = 7
            else                             : v = -1
        elif     (10752 <= t) and (t < 10816):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 10752): v = 10
            else                             : v = -1
        elif     (10816 <= t) and (t < 10880):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 10752): v = 10
            elif (10752 <= s) and (s < 10816): v = 7
            else                             : v = -1
        elif     (10880 <= t) and (t < 10944):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 10752): v = 10
            elif (10752 <= s) and (s < 10880): v = 8
            else                             : v = -1
        elif     (10944 <= t) and (t < 11008):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 10752): v = 10
            elif (10752 <= s) and (s < 10880): v = 8
            elif (10880 <= s) and (s < 10944): v = 7
            else                             : v = -1
        elif     (11008 <= t) and (t < 11072):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 10752): v = 10
            elif (10752 <= s) and (s < 11008): v = 9
            else                             : v = -1
        elif     (11072 <= t) and (t < 11136):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 10752): v = 10
            elif (10752 <= s) and (s < 11008): v = 9
            elif (11008 <= s) and (s < 11072): v = 7
            else                             : v = -1
        elif     (11136 <= t) and (t < 11200):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 10752): v = 10
            elif (10752 <= s) and (s < 11008): v = 9
            elif (11008 <= s) and (s < 11136): v = 8
            else                             : v = -1
        elif     (11200 <= t) and (t < 11264):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 10752): v = 10
            elif (10752 <= s) and (s < 11008): v = 9
            elif (11008 <= s) and (s < 11136): v = 8
            elif (11136 <= s) and (s < 11200): v = 7
            else                             : v = -1
        elif     (11264 <= t) and (t < 11328):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 11264): v = 11
            else                             : v = -1
        elif     (11328 <= t) and (t < 11392):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 11264): v = 11
            elif (11264 <= s) and (s < 11328): v = 7
            else                             : v = -1
        elif     (11392 <= t) and (t < 11456):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 11264): v = 11
            elif (11264 <= s) and (s < 11392): v = 8
            else                             : v = -1
        elif     (11456 <= t) and (t < 11520):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 11264): v = 11
            elif (11264 <= s) and (s < 11392): v = 8
            elif (11392 <= s) and (s < 11456): v = 7
            else                             : v = -1
        elif     (11520 <= t) and (t < 11584):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 11264): v = 11
            elif (11264 <= s) and (s < 11520): v = 9
            else                             : v = -1
        elif     (11584 <= t) and (t < 11648):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 11264): v = 11
            elif (11264 <= s) and (s < 11520): v = 9
            elif (11520 <= s) and (s < 11584): v = 7
            else                             : v = -1
        elif     (11648 <= t) and (t < 11712):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 11264): v = 11
            elif (11264 <= s) and (s < 11520): v = 9
            elif (11520 <= s) and (s < 11648): v = 8
            else                             : v = -1
        elif     (11712 <= t) and (t < 11776):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 11264): v = 11
            elif (11264 <= s) and (s < 11520): v = 9
            elif (11520 <= s) and (s < 11648): v = 8
            elif (11648 <= s) and (s < 11712): v = 7
            else                             : v = -1
        elif     (11776 <= t) and (t < 11840):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 11264): v = 11
            elif (11264 <= s) and (s < 11776): v = 10
            else                             : v = -1
        elif     (11840 <= t) and (t < 11904):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 11264): v = 11
            elif (11264 <= s) and (s < 11776): v = 10
            elif (11776 <= s) and (s < 11840): v = 7
            else                             : v = -1
        elif     (11904 <= t) and (t < 11968):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 11264): v = 11
            elif (11264 <= s) and (s < 11776): v = 10
            elif (11776 <= s) and (s < 11904): v = 8
            else                             : v = -1
        elif     (11968 <= t) and (t < 12032):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 11264): v = 11
            elif (11264 <= s) and (s < 11776): v = 10
            elif (11776 <= s) and (s < 11904): v = 8
            elif (11904 <= s) and (s < 11968): v = 7
            else                             : v = -1
        elif     (12032 <= t) and (t < 12096):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 11264): v = 11
            elif (11264 <= s) and (s < 11776): v = 10
            elif (11776 <= s) and (s < 12032): v = 9
            else                             : v = -1
        elif     (12096 <= t) and (t < 12160):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 11264): v = 11
            elif (11264 <= s) and (s < 11776): v = 10
            elif (11776 <= s) and (s < 12032): v = 9
            elif (12032 <= s) and (s < 12096): v = 7
            else                             : v = -1
        elif     (12160 <= t) and (t < 12224):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 11264): v = 11
            elif (11264 <= s) and (s < 11776): v = 10
            elif (11776 <= s) and (s < 12032): v = 9
            elif (12032 <= s) and (s < 12160): v = 8
            else                             : v = -1
        elif     (12224 <= t) and (t < 12288):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 11264): v = 11
            elif (11264 <= s) and (s < 11776): v = 10
            elif (11776 <= s) and (s < 12032): v = 9
            elif (12032 <= s) and (s < 12160): v = 8
            elif (12160 <= s) and (s < 12224): v = 7
            else                             : v = -1
        elif     (12288 <= t) and (t < 12352):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            else                             : v = -1
        elif     (12352 <= t) and (t < 12416):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 12352): v = 7
            else                             : v = -1
        elif     (12416 <= t) and (t < 12480):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 12416): v = 8
            else                             : v = -1
        elif     (12480 <= t) and (t < 12544):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 12416): v = 8
            elif (12416 <= s) and (s < 12480): v = 7
            else                             : v = -1
        elif     (12544 <= t) and (t < 12608):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 12544): v = 9
            else                             : v = -1
        elif     (12608 <= t) and (t < 12672):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 12544): v = 9
            elif (12544 <= s) and (s < 12608): v = 7
            else                             : v = -1
        elif     (12672 <= t) and (t < 12736):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 12544): v = 9
            elif (12544 <= s) and (s < 12672): v = 8
            else                             : v = -1
        elif     (12736 <= t) and (t < 12800):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 12544): v = 9
            elif (12544 <= s) and (s < 12672): v = 8
            elif (12672 <= s) and (s < 12736): v = 7
            else                             : v = -1
        elif     (12800 <= t) and (t < 12864):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 12800): v = 10
            else                             : v = -1
        elif     (12864 <= t) and (t < 12928):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 12800): v = 10
            elif (12800 <= s) and (s < 12864): v = 7
            else                             : v = -1
        elif     (12928 <= t) and (t < 12992):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 12800): v = 10
            elif (12800 <= s) and (s < 12928): v = 8
            else                             : v = -1
        elif     (12992 <= t) and (t < 13056):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 12800): v = 10
            elif (12800 <= s) and (s < 12928): v = 8
            elif (12928 <= s) and (s < 12992): v = 7
            else                             : v = -1
        elif     (13056 <= t) and (t < 13120):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 12800): v = 10
            elif (12800 <= s) and (s < 13056): v = 9
            else                             : v = -1
        elif     (13120 <= t) and (t < 13184):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 12800): v = 10
            elif (12800 <= s) and (s < 13056): v = 9
            elif (13056 <= s) and (s < 13120): v = 7
            else                             : v = -1
        elif     (13184 <= t) and (t < 13248):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 12800): v = 10
            elif (12800 <= s) and (s < 13056): v = 9
            elif (13056 <= s) and (s < 13184): v = 8
            else                             : v = -1
        elif     (13248 <= t) and (t < 13312):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 12800): v = 10
            elif (12800 <= s) and (s < 13056): v = 9
            elif (13056 <= s) and (s < 13184): v = 8
            elif (13184 <= s) and (s < 13248): v = 7
            else                             : v = -1
        elif     (13312 <= t) and (t < 13376):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 13312): v = 11
            else                             : v = -1
        elif     (13376 <= t) and (t < 13440):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 13312): v = 11
            elif (13312 <= s) and (s < 13376): v = 7
            else                             : v = -1
        elif     (13440 <= t) and (t < 13504):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 13312): v = 11
            elif (13312 <= s) and (s < 13440): v = 8
            else                             : v = -1
        elif     (13504 <= t) and (t < 13568):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 13312): v = 11
            elif (13312 <= s) and (s < 13440): v = 8
            elif (13440 <= s) and (s < 13504): v = 7
            else                             : v = -1
        elif     (13568 <= t) and (t < 13632):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 13312): v = 11
            elif (13312 <= s) and (s < 13568): v = 9
            else                             : v = -1
        elif     (13632 <= t) and (t < 13696):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 13312): v = 11
            elif (13312 <= s) and (s < 13568): v = 9
            elif (13568 <= s) and (s < 13632): v = 7
            else                             : v = -1
        elif     (13696 <= t) and (t < 13760):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 13312): v = 11
            elif (13312 <= s) and (s < 13568): v = 9
            elif (13568 <= s) and (s < 13696): v = 8
            else                             : v = -1
        elif     (13760 <= t) and (t < 13824):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 13312): v = 11
            elif (13312 <= s) and (s < 13568): v = 9
            elif (13568 <= s) and (s < 13696): v = 8
            elif (13696 <= s) and (s < 13760): v = 7
            else                             : v = -1
        elif     (13824 <= t) and (t < 13888):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 13312): v = 11
            elif (13312 <= s) and (s < 13824): v = 10
            else                             : v = -1
        elif     (13888 <= t) and (t < 13952):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 13312): v = 11
            elif (13312 <= s) and (s < 13824): v = 10
            elif (13824 <= s) and (s < 13888): v = 7
            else                             : v = -1
        elif     (13952 <= t) and (t < 14016):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 13312): v = 11
            elif (13312 <= s) and (s < 13824): v = 10
            elif (13824 <= s) and (s < 13952): v = 8
            else                             : v = -1
        elif     (14016 <= t) and (t < 14080):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 13312): v = 11
            elif (13312 <= s) and (s < 13824): v = 10
            elif (13824 <= s) and (s < 13952): v = 8
            elif (13952 <= s) and (s < 14016): v = 7
            else                             : v = -1
        elif     (14080 <= t) and (t < 14144):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 13312): v = 11
            elif (13312 <= s) and (s < 13824): v = 10
            elif (13824 <= s) and (s < 14080): v = 9
            else                             : v = -1
        elif     (14144 <= t) and (t < 14208):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 13312): v = 11
            elif (13312 <= s) and (s < 13824): v = 10
            elif (13824 <= s) and (s < 14080): v = 9
            elif (14080 <= s) and (s < 14144): v = 7
            else                             : v = -1
        elif     (14208 <= t) and (t < 14272):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 13312): v = 11
            elif (13312 <= s) and (s < 13824): v = 10
            elif (13824 <= s) and (s < 14080): v = 9
            elif (14080 <= s) and (s < 14208): v = 8
            else                             : v = -1
        elif     (14272 <= t) and (t < 14336):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 13312): v = 11
            elif (13312 <= s) and (s < 13824): v = 10
            elif (13824 <= s) and (s < 14080): v = 9
            elif (14080 <= s) and (s < 14208): v = 8
            elif (14208 <= s) and (s < 14272): v = 7
            else                             : v = -1
        elif     (14336 <= t) and (t < 14400):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            else                             : v = -1
        elif     (14400 <= t) and (t < 14464):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 14400): v = 7
            else                             : v = -1
        elif     (14464 <= t) and (t < 14528):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 14464): v = 8
            else                             : v = -1
        elif     (14528 <= t) and (t < 14592):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 14464): v = 8
            elif (14464 <= s) and (s < 14528): v = 7
            else                             : v = -1
        elif     (14592 <= t) and (t < 14656):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 14592): v = 9
            else                             : v = -1
        elif     (14656 <= t) and (t < 14720):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 14592): v = 9
            elif (14592 <= s) and (s < 14656): v = 7
            else                             : v = -1
        elif     (14720 <= t) and (t < 14784):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 14592): v = 9
            elif (14592 <= s) and (s < 14720): v = 8
            else                             : v = -1
        elif     (14784 <= t) and (t < 14848):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 14592): v = 9
            elif (14592 <= s) and (s < 14720): v = 8
            elif (14720 <= s) and (s < 14784): v = 7
            else                             : v = -1
        elif     (14848 <= t) and (t < 14912):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 14848): v = 10
            else                             : v = -1
        elif     (14912 <= t) and (t < 14976):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 14848): v = 10
            elif (14848 <= s) and (s < 14912): v = 7
            else                             : v = -1
        elif     (14976 <= t) and (t < 15040):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 14848): v = 10
            elif (14848 <= s) and (s < 14976): v = 8
            else                             : v = -1
        elif     (15040 <= t) and (t < 15104):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 14848): v = 10
            elif (14848 <= s) and (s < 14976): v = 8
            elif (14976 <= s) and (s < 15040): v = 7
            else                             : v = -1
        elif     (15104 <= t) and (t < 15168):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 14848): v = 10
            elif (14848 <= s) and (s < 15104): v = 9
            else                             : v = -1
        elif     (15168 <= t) and (t < 15232):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 14848): v = 10
            elif (14848 <= s) and (s < 15104): v = 9
            elif (15104 <= s) and (s < 15168): v = 7
            else                             : v = -1
        elif     (15232 <= t) and (t < 15296):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 14848): v = 10
            elif (14848 <= s) and (s < 15104): v = 9
            elif (15104 <= s) and (s < 15232): v = 8
            else                             : v = -1
        elif     (15296 <= t) and (t < 15360):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 14848): v = 10
            elif (14848 <= s) and (s < 15104): v = 9
            elif (15104 <= s) and (s < 15232): v = 8
            elif (15232 <= s) and (s < 15296): v = 7
            else                             : v = -1
        elif     (15360 <= t) and (t < 15424):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 15360): v = 11
            else                             : v = -1
        elif     (15424 <= t) and (t < 15488):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 15360): v = 11
            elif (15360 <= s) and (s < 15424): v = 7
            else                             : v = -1
        elif     (15488 <= t) and (t < 15552):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 15360): v = 11
            elif (15360 <= s) and (s < 15488): v = 8
            else                             : v = -1
        elif     (15552 <= t) and (t < 15616):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 15360): v = 11
            elif (15360 <= s) and (s < 15488): v = 8
            elif (15488 <= s) and (s < 15552): v = 7
            else                             : v = -1
        elif     (15616 <= t) and (t < 15680):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 15360): v = 11
            elif (15360 <= s) and (s < 15616): v = 9
            else                             : v = -1
        elif     (15680 <= t) and (t < 15744):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 15360): v = 11
            elif (15360 <= s) and (s < 15616): v = 9
            elif (15616 <= s) and (s < 15680): v = 7
            else                             : v = -1
        elif     (15744 <= t) and (t < 15808):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 15360): v = 11
            elif (15360 <= s) and (s < 15616): v = 9
            elif (15616 <= s) and (s < 15744): v = 8
            else                             : v = -1
        elif     (15808 <= t) and (t < 15872):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 15360): v = 11
            elif (15360 <= s) and (s < 15616): v = 9
            elif (15616 <= s) and (s < 15744): v = 8
            elif (15744 <= s) and (s < 15808): v = 7
            else                             : v = -1
        elif     (15872 <= t) and (t < 15936):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 15360): v = 11
            elif (15360 <= s) and (s < 15872): v = 10
            else                             : v = -1
        elif     (15936 <= t) and (t < 16000):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 15360): v = 11
            elif (15360 <= s) and (s < 15872): v = 10
            elif (15872 <= s) and (s < 15936): v = 7
            else                             : v = -1
        elif     (16000 <= t) and (t < 16064):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 15360): v = 11
            elif (15360 <= s) and (s < 15872): v = 10
            elif (15872 <= s) and (s < 16000): v = 8
            else                             : v = -1
        elif     (16064 <= t) and (t < 16128):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 15360): v = 11
            elif (15360 <= s) and (s < 15872): v = 10
            elif (15872 <= s) and (s < 16000): v = 8
            elif (16000 <= s) and (s < 16064): v = 7
            else                             : v = -1
        elif     (16128 <= t) and (t < 16192):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 15360): v = 11
            elif (15360 <= s) and (s < 15872): v = 10
            elif (15872 <= s) and (s < 16128): v = 9
            else                             : v = -1
        elif     (16192 <= t) and (t < 16256):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 15360): v = 11
            elif (15360 <= s) and (s < 15872): v = 10
            elif (15872 <= s) and (s < 16128): v = 9
            elif (16128 <= s) and (s < 16192): v = 7
            else                             : v = -1
        elif     (16256 <= t) and (t < 16320):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 15360): v = 11
            elif (15360 <= s) and (s < 15872): v = 10
            elif (15872 <= s) and (s < 16128): v = 9
            elif (16128 <= s) and (s < 16256): v = 8
            else                             : v = -1
        elif     (16320 <= t) and (t < 16384):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 15360): v = 11
            elif (15360 <= s) and (s < 15872): v = 10
            elif (15872 <= s) and (s < 16128): v = 9
            elif (16128 <= s) and (s < 16256): v = 8
            elif (16256 <= s) and (s < 16320): v = 7
            else                             : v = -1
        else                                 : v = -1
    if (LB == 2 and (BT == 128 and BS == 128)):
        if       (0     <= t) and (t < 128  ): v = -1
        elif     (128   <= t) and (t < 256  ):
            if   (0     <= s) and (s < 128  ): v = 8
            else                             : v = -1
        elif     (256   <= t) and (t < 384  ):
            if   (0     <= s) and (s < 256  ): v = 9
            else                             : v = -1
        elif     (384   <= t) and (t < 512  ):
            if   (0     <= s) and (s < 256  ): v = 9
            elif (256   <= s) and (s < 384  ): v = 8
            else                             : v = -1
        elif     (512   <= t) and (t < 640  ):
            if   (0     <= s) and (s < 512  ): v = 10
            else                             : v = -1
        elif     (640   <= t) and (t < 768  ):
            if   (0     <= s) and (s < 512  ): v = 10
            elif (512   <= s) and (s < 640  ): v = 8
            else                             : v = -1
        elif     (768   <= t) and (t < 896  ):
            if   (0     <= s) and (s < 512  ): v = 10
            elif (512   <= s) and (s < 768  ): v = 9
            else                             : v = -1
        elif     (896   <= t) and (t < 1024 ):
            if   (0     <= s) and (s < 512  ): v = 10
            elif (512   <= s) and (s < 768  ): v = 9
            elif (768   <= s) and (s < 896  ): v = 8
            else                             : v = -1
        elif     (1024  <= t) and (t < 1152 ):
            if   (0     <= s) and (s < 1024 ): v = 11
            else                             : v = -1
        elif     (1152  <= t) and (t < 1280 ):
            if   (0     <= s) and (s < 1024 ): v = 11
            elif (1024  <= s) and (s < 1152 ): v = 8
            else                             : v = -1
        elif     (1280  <= t) and (t < 1408 ):
            if   (0     <= s) and (s < 1024 ): v = 11
            elif (1024  <= s) and (s < 1280 ): v = 9
            else                             : v = -1
        elif     (1408  <= t) and (t < 1536 ):
            if   (0     <= s) and (s < 1024 ): v = 11
            elif (1024  <= s) and (s < 1280 ): v = 9
            elif (1280  <= s) and (s < 1408 ): v = 8
            else                             : v = -1
        elif     (1536  <= t) and (t < 1664 ):
            if   (0     <= s) and (s < 1024 ): v = 11
            elif (1024  <= s) and (s < 1536 ): v = 10
            else                             : v = -1
        elif     (1664  <= t) and (t < 1792 ):
            if   (0     <= s) and (s < 1024 ): v = 11
            elif (1024  <= s) and (s < 1536 ): v = 10
            elif (1536  <= s) and (s < 1664 ): v = 8
            else                             : v = -1
        elif     (1792  <= t) and (t < 1920 ):
            if   (0     <= s) and (s < 1024 ): v = 11
            elif (1024  <= s) and (s < 1536 ): v = 10
            elif (1536  <= s) and (s < 1792 ): v = 9
            else                             : v = -1
        elif     (1920  <= t) and (t < 2048 ):
            if   (0     <= s) and (s < 1024 ): v = 11
            elif (1024  <= s) and (s < 1536 ): v = 10
            elif (1536  <= s) and (s < 1792 ): v = 9
            elif (1792  <= s) and (s < 1920 ): v = 8
            else                             : v = -1
        elif     (2048  <= t) and (t < 2176 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            else                             : v = -1
        elif     (2176  <= t) and (t < 2304 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 2176 ): v = 8
            else                             : v = -1
        elif     (2304  <= t) and (t < 2432 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 2304 ): v = 9
            else                             : v = -1
        elif     (2432  <= t) and (t < 2560 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 2304 ): v = 9
            elif (2304  <= s) and (s < 2432 ): v = 8
            else                             : v = -1
        elif     (2560  <= t) and (t < 2688 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 2560 ): v = 10
            else                             : v = -1
        elif     (2688  <= t) and (t < 2816 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 2560 ): v = 10
            elif (2560  <= s) and (s < 2688 ): v = 8
            else                             : v = -1
        elif     (2816  <= t) and (t < 2944 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 2560 ): v = 10
            elif (2560  <= s) and (s < 2816 ): v = 9
            else                             : v = -1
        elif     (2944  <= t) and (t < 3072 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 2560 ): v = 10
            elif (2560  <= s) and (s < 2816 ): v = 9
            elif (2816  <= s) and (s < 2944 ): v = 8
            else                             : v = -1
        elif     (3072  <= t) and (t < 3200 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 3072 ): v = 11
            else                             : v = -1
        elif     (3200  <= t) and (t < 3328 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 3072 ): v = 11
            elif (3072  <= s) and (s < 3200 ): v = 8
            else                             : v = -1
        elif     (3328  <= t) and (t < 3456 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 3072 ): v = 11
            elif (3072  <= s) and (s < 3328 ): v = 9
            else                             : v = -1
        elif     (3456  <= t) and (t < 3584 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 3072 ): v = 11
            elif (3072  <= s) and (s < 3328 ): v = 9
            elif (3328  <= s) and (s < 3456 ): v = 8
            else                             : v = -1
        elif     (3584  <= t) and (t < 3712 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 3072 ): v = 11
            elif (3072  <= s) and (s < 3584 ): v = 10
            else                             : v = -1
        elif     (3712  <= t) and (t < 3840 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 3072 ): v = 11
            elif (3072  <= s) and (s < 3584 ): v = 10
            elif (3584  <= s) and (s < 3712 ): v = 8
            else                             : v = -1
        elif     (3840  <= t) and (t < 3968 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 3072 ): v = 11
            elif (3072  <= s) and (s < 3584 ): v = 10
            elif (3584  <= s) and (s < 3840 ): v = 9
            else                             : v = -1
        elif     (3968  <= t) and (t < 4096 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 3072 ): v = 11
            elif (3072  <= s) and (s < 3584 ): v = 10
            elif (3584  <= s) and (s < 3840 ): v = 9
            elif (3840  <= s) and (s < 3968 ): v = 8
            else                             : v = -1
        elif     (4096  <= t) and (t < 4224 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            else                             : v = -1
        elif     (4224  <= t) and (t < 4352 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 4224 ): v = 8
            else                             : v = -1
        elif     (4352  <= t) and (t < 4480 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 4352 ): v = 9
            else                             : v = -1
        elif     (4480  <= t) and (t < 4608 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 4352 ): v = 9
            elif (4352  <= s) and (s < 4480 ): v = 8
            else                             : v = -1
        elif     (4608  <= t) and (t < 4736 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 4608 ): v = 10
            else                             : v = -1
        elif     (4736  <= t) and (t < 4864 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 4608 ): v = 10
            elif (4608  <= s) and (s < 4736 ): v = 8
            else                             : v = -1
        elif     (4864  <= t) and (t < 4992 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 4608 ): v = 10
            elif (4608  <= s) and (s < 4864 ): v = 9
            else                             : v = -1
        elif     (4992  <= t) and (t < 5120 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 4608 ): v = 10
            elif (4608  <= s) and (s < 4864 ): v = 9
            elif (4864  <= s) and (s < 4992 ): v = 8
            else                             : v = -1
        elif     (5120  <= t) and (t < 5248 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 5120 ): v = 11
            else                             : v = -1
        elif     (5248  <= t) and (t < 5376 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 5120 ): v = 11
            elif (5120  <= s) and (s < 5248 ): v = 8
            else                             : v = -1
        elif     (5376  <= t) and (t < 5504 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 5120 ): v = 11
            elif (5120  <= s) and (s < 5376 ): v = 9
            else                             : v = -1
        elif     (5504  <= t) and (t < 5632 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 5120 ): v = 11
            elif (5120  <= s) and (s < 5376 ): v = 9
            elif (5376  <= s) and (s < 5504 ): v = 8
            else                             : v = -1
        elif     (5632  <= t) and (t < 5760 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 5120 ): v = 11
            elif (5120  <= s) and (s < 5632 ): v = 10
            else                             : v = -1
        elif     (5760  <= t) and (t < 5888 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 5120 ): v = 11
            elif (5120  <= s) and (s < 5632 ): v = 10
            elif (5632  <= s) and (s < 5760 ): v = 8
            else                             : v = -1
        elif     (5888  <= t) and (t < 6016 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 5120 ): v = 11
            elif (5120  <= s) and (s < 5632 ): v = 10
            elif (5632  <= s) and (s < 5888 ): v = 9
            else                             : v = -1
        elif     (6016  <= t) and (t < 6144 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 5120 ): v = 11
            elif (5120  <= s) and (s < 5632 ): v = 10
            elif (5632  <= s) and (s < 5888 ): v = 9
            elif (5888  <= s) and (s < 6016 ): v = 8
            else                             : v = -1
        elif     (6144  <= t) and (t < 6272 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            else                             : v = -1
        elif     (6272  <= t) and (t < 6400 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 6272 ): v = 8
            else                             : v = -1
        elif     (6400  <= t) and (t < 6528 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 6400 ): v = 9
            else                             : v = -1
        elif     (6528  <= t) and (t < 6656 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 6400 ): v = 9
            elif (6400  <= s) and (s < 6528 ): v = 8
            else                             : v = -1
        elif     (6656  <= t) and (t < 6784 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 6656 ): v = 10
            else                             : v = -1
        elif     (6784  <= t) and (t < 6912 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 6656 ): v = 10
            elif (6656  <= s) and (s < 6784 ): v = 8
            else                             : v = -1
        elif     (6912  <= t) and (t < 7040 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 6656 ): v = 10
            elif (6656  <= s) and (s < 6912 ): v = 9
            else                             : v = -1
        elif     (7040  <= t) and (t < 7168 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 6656 ): v = 10
            elif (6656  <= s) and (s < 6912 ): v = 9
            elif (6912  <= s) and (s < 7040 ): v = 8
            else                             : v = -1
        elif     (7168  <= t) and (t < 7296 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 7168 ): v = 11
            else                             : v = -1
        elif     (7296  <= t) and (t < 7424 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 7168 ): v = 11
            elif (7168  <= s) and (s < 7296 ): v = 8
            else                             : v = -1
        elif     (7424  <= t) and (t < 7552 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 7168 ): v = 11
            elif (7168  <= s) and (s < 7424 ): v = 9
            else                             : v = -1
        elif     (7552  <= t) and (t < 7680 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 7168 ): v = 11
            elif (7168  <= s) and (s < 7424 ): v = 9
            elif (7424  <= s) and (s < 7552 ): v = 8
            else                             : v = -1
        elif     (7680  <= t) and (t < 7808 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 7168 ): v = 11
            elif (7168  <= s) and (s < 7680 ): v = 10
            else                             : v = -1
        elif     (7808  <= t) and (t < 7936 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 7168 ): v = 11
            elif (7168  <= s) and (s < 7680 ): v = 10
            elif (7680  <= s) and (s < 7808 ): v = 8
            else                             : v = -1
        elif     (7936  <= t) and (t < 8064 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 7168 ): v = 11
            elif (7168  <= s) and (s < 7680 ): v = 10
            elif (7680  <= s) and (s < 7936 ): v = 9
            else                             : v = -1
        elif     (8064  <= t) and (t < 8192 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 7168 ): v = 11
            elif (7168  <= s) and (s < 7680 ): v = 10
            elif (7680  <= s) and (s < 7936 ): v = 9
            elif (7936  <= s) and (s < 8064 ): v = 8
            else                             : v = -1
        elif     (8192  <= t) and (t < 8320 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            else                             : v = -1
        elif     (8320  <= t) and (t < 8448 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 8320 ): v = 8
            else                             : v = -1
        elif     (8448  <= t) and (t < 8576 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 8448 ): v = 9
            else                             : v = -1
        elif     (8576  <= t) and (t < 8704 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 8448 ): v = 9
            elif (8448  <= s) and (s < 8576 ): v = 8
            else                             : v = -1
        elif     (8704  <= t) and (t < 8832 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 8704 ): v = 10
            else                             : v = -1
        elif     (8832  <= t) and (t < 8960 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 8704 ): v = 10
            elif (8704  <= s) and (s < 8832 ): v = 8
            else                             : v = -1
        elif     (8960  <= t) and (t < 9088 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 8704 ): v = 10
            elif (8704  <= s) and (s < 8960 ): v = 9
            else                             : v = -1
        elif     (9088  <= t) and (t < 9216 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 8704 ): v = 10
            elif (8704  <= s) and (s < 8960 ): v = 9
            elif (8960  <= s) and (s < 9088 ): v = 8
            else                             : v = -1
        elif     (9216  <= t) and (t < 9344 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 9216 ): v = 11
            else                             : v = -1
        elif     (9344  <= t) and (t < 9472 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 9216 ): v = 11
            elif (9216  <= s) and (s < 9344 ): v = 8
            else                             : v = -1
        elif     (9472  <= t) and (t < 9600 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 9216 ): v = 11
            elif (9216  <= s) and (s < 9472 ): v = 9
            else                             : v = -1
        elif     (9600  <= t) and (t < 9728 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 9216 ): v = 11
            elif (9216  <= s) and (s < 9472 ): v = 9
            elif (9472  <= s) and (s < 9600 ): v = 8
            else                             : v = -1
        elif     (9728  <= t) and (t < 9856 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 9216 ): v = 11
            elif (9216  <= s) and (s < 9728 ): v = 10
            else                             : v = -1
        elif     (9856  <= t) and (t < 9984 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 9216 ): v = 11
            elif (9216  <= s) and (s < 9728 ): v = 10
            elif (9728  <= s) and (s < 9856 ): v = 8
            else                             : v = -1
        elif     (9984  <= t) and (t < 10112):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 9216 ): v = 11
            elif (9216  <= s) and (s < 9728 ): v = 10
            elif (9728  <= s) and (s < 9984 ): v = 9
            else                             : v = -1
        elif     (10112 <= t) and (t < 10240):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 9216 ): v = 11
            elif (9216  <= s) and (s < 9728 ): v = 10
            elif (9728  <= s) and (s < 9984 ): v = 9
            elif (9984  <= s) and (s < 10112): v = 8
            else                             : v = -1
        elif     (10240 <= t) and (t < 10368):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            else                             : v = -1
        elif     (10368 <= t) and (t < 10496):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 10368): v = 8
            else                             : v = -1
        elif     (10496 <= t) and (t < 10624):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 10496): v = 9
            else                             : v = -1
        elif     (10624 <= t) and (t < 10752):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 10496): v = 9
            elif (10496 <= s) and (s < 10624): v = 8
            else                             : v = -1
        elif     (10752 <= t) and (t < 10880):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 10752): v = 10
            else                             : v = -1
        elif     (10880 <= t) and (t < 11008):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 10752): v = 10
            elif (10752 <= s) and (s < 10880): v = 8
            else                             : v = -1
        elif     (11008 <= t) and (t < 11136):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 10752): v = 10
            elif (10752 <= s) and (s < 11008): v = 9
            else                             : v = -1
        elif     (11136 <= t) and (t < 11264):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 10752): v = 10
            elif (10752 <= s) and (s < 11008): v = 9
            elif (11008 <= s) and (s < 11136): v = 8
            else                             : v = -1
        elif     (11264 <= t) and (t < 11392):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 11264): v = 11
            else                             : v = -1
        elif     (11392 <= t) and (t < 11520):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 11264): v = 11
            elif (11264 <= s) and (s < 11392): v = 8
            else                             : v = -1
        elif     (11520 <= t) and (t < 11648):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 11264): v = 11
            elif (11264 <= s) and (s < 11520): v = 9
            else                             : v = -1
        elif     (11648 <= t) and (t < 11776):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 11264): v = 11
            elif (11264 <= s) and (s < 11520): v = 9
            elif (11520 <= s) and (s < 11648): v = 8
            else                             : v = -1
        elif     (11776 <= t) and (t < 11904):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 11264): v = 11
            elif (11264 <= s) and (s < 11776): v = 10
            else                             : v = -1
        elif     (11904 <= t) and (t < 12032):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 11264): v = 11
            elif (11264 <= s) and (s < 11776): v = 10
            elif (11776 <= s) and (s < 11904): v = 8
            else                             : v = -1
        elif     (12032 <= t) and (t < 12160):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 11264): v = 11
            elif (11264 <= s) and (s < 11776): v = 10
            elif (11776 <= s) and (s < 12032): v = 9
            else                             : v = -1
        elif     (12160 <= t) and (t < 12288):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 11264): v = 11
            elif (11264 <= s) and (s < 11776): v = 10
            elif (11776 <= s) and (s < 12032): v = 9
            elif (12032 <= s) and (s < 12160): v = 8
            else                             : v = -1
        elif     (12288 <= t) and (t < 12416):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            else                             : v = -1
        elif     (12416 <= t) and (t < 12544):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 12416): v = 8
            else                             : v = -1
        elif     (12544 <= t) and (t < 12672):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 12544): v = 9
            else                             : v = -1
        elif     (12672 <= t) and (t < 12800):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 12544): v = 9
            elif (12544 <= s) and (s < 12672): v = 8
            else                             : v = -1
        elif     (12800 <= t) and (t < 12928):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 12800): v = 10
            else                             : v = -1
        elif     (12928 <= t) and (t < 13056):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 12800): v = 10
            elif (12800 <= s) and (s < 12928): v = 8
            else                             : v = -1
        elif     (13056 <= t) and (t < 13184):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 12800): v = 10
            elif (12800 <= s) and (s < 13056): v = 9
            else                             : v = -1
        elif     (13184 <= t) and (t < 13312):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 12800): v = 10
            elif (12800 <= s) and (s < 13056): v = 9
            elif (13056 <= s) and (s < 13184): v = 8
            else                             : v = -1
        elif     (13312 <= t) and (t < 13440):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 13312): v = 11
            else                             : v = -1
        elif     (13440 <= t) and (t < 13568):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 13312): v = 11
            elif (13312 <= s) and (s < 13440): v = 8
            else                             : v = -1
        elif     (13568 <= t) and (t < 13696):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 13312): v = 11
            elif (13312 <= s) and (s < 13568): v = 9
            else                             : v = -1
        elif     (13696 <= t) and (t < 13824):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 13312): v = 11
            elif (13312 <= s) and (s < 13568): v = 9
            elif (13568 <= s) and (s < 13696): v = 8
            else                             : v = -1
        elif     (13824 <= t) and (t < 13952):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 13312): v = 11
            elif (13312 <= s) and (s < 13824): v = 10
            else                             : v = -1
        elif     (13952 <= t) and (t < 14080):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 13312): v = 11
            elif (13312 <= s) and (s < 13824): v = 10
            elif (13824 <= s) and (s < 13952): v = 8
            else                             : v = -1
        elif     (14080 <= t) and (t < 14208):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 13312): v = 11
            elif (13312 <= s) and (s < 13824): v = 10
            elif (13824 <= s) and (s < 14080): v = 9
            else                             : v = -1
        elif     (14208 <= t) and (t < 14336):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 13312): v = 11
            elif (13312 <= s) and (s < 13824): v = 10
            elif (13824 <= s) and (s < 14080): v = 9
            elif (14080 <= s) and (s < 14208): v = 8
            else                             : v = -1
        elif     (14336 <= t) and (t < 14464):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            else                             : v = -1
        elif     (14464 <= t) and (t < 14592):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 14464): v = 8
            else                             : v = -1
        elif     (14592 <= t) and (t < 14720):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 14592): v = 9
            else                             : v = -1
        elif     (14720 <= t) and (t < 14848):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 14592): v = 9
            elif (14592 <= s) and (s < 14720): v = 8
            else                             : v = -1
        elif     (14848 <= t) and (t < 14976):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 14848): v = 10
            else                             : v = -1
        elif     (14976 <= t) and (t < 15104):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 14848): v = 10
            elif (14848 <= s) and (s < 14976): v = 8
            else                             : v = -1
        elif     (15104 <= t) and (t < 15232):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 14848): v = 10
            elif (14848 <= s) and (s < 15104): v = 9
            else                             : v = -1
        elif     (15232 <= t) and (t < 15360):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 14848): v = 10
            elif (14848 <= s) and (s < 15104): v = 9
            elif (15104 <= s) and (s < 15232): v = 8
            else                             : v = -1
        elif     (15360 <= t) and (t < 15488):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 15360): v = 11
            else                             : v = -1
        elif     (15488 <= t) and (t < 15616):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 15360): v = 11
            elif (15360 <= s) and (s < 15488): v = 8
            else                             : v = -1
        elif     (15616 <= t) and (t < 15744):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 15360): v = 11
            elif (15360 <= s) and (s < 15616): v = 9
            else                             : v = -1
        elif     (15744 <= t) and (t < 15872):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 15360): v = 11
            elif (15360 <= s) and (s < 15616): v = 9
            elif (15616 <= s) and (s < 15744): v = 8
            else                             : v = -1
        elif     (15872 <= t) and (t < 16000):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 15360): v = 11
            elif (15360 <= s) and (s < 15872): v = 10
            else                             : v = -1
        elif     (16000 <= t) and (t < 16128):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 15360): v = 11
            elif (15360 <= s) and (s < 15872): v = 10
            elif (15872 <= s) and (s < 16000): v = 8
            else                             : v = -1
        elif     (16128 <= t) and (t < 16256):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 15360): v = 11
            elif (15360 <= s) and (s < 15872): v = 10
            elif (15872 <= s) and (s < 16128): v = 9
            else                             : v = -1
        elif     (16256 <= t) and (t < 16384):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 15360): v = 11
            elif (15360 <= s) and (s < 15872): v = 10
            elif (15872 <= s) and (s < 16128): v = 9
            elif (16128 <= s) and (s < 16256): v = 8
            else                             : v = -1
        else                                 : v = -1
    if (LB == 2 and (BT == 256 and BS == 256)):
        if       (0     <= t) and (t < 256  ): v = -1
        elif     (256   <= t) and (t < 512  ):
            if   (0     <= s) and (s < 256  ): v = 9
            else                             : v = -1
        elif     (512   <= t) and (t < 768  ):
            if   (0     <= s) and (s < 512  ): v = 10
            else                             : v = -1
        elif     (768   <= t) and (t < 1024 ):
            if   (0     <= s) and (s < 512  ): v = 10
            elif (512   <= s) and (s < 768  ): v = 9
            else                             : v = -1
        elif     (1024  <= t) and (t < 1280 ):
            if   (0     <= s) and (s < 1024 ): v = 11
            else                             : v = -1
        elif     (1280  <= t) and (t < 1536 ):
            if   (0     <= s) and (s < 1024 ): v = 11
            elif (1024  <= s) and (s < 1280 ): v = 9
            else                             : v = -1
        elif     (1536  <= t) and (t < 1792 ):
            if   (0     <= s) and (s < 1024 ): v = 11
            elif (1024  <= s) and (s < 1536 ): v = 10
            else                             : v = -1
        elif     (1792  <= t) and (t < 2048 ):
            if   (0     <= s) and (s < 1024 ): v = 11
            elif (1024  <= s) and (s < 1536 ): v = 10
            elif (1536  <= s) and (s < 1792 ): v = 9
            else                             : v = -1
        elif     (2048  <= t) and (t < 2304 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            else                             : v = -1
        elif     (2304  <= t) and (t < 2560 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 2304 ): v = 9
            else                             : v = -1
        elif     (2560  <= t) and (t < 2816 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 2560 ): v = 10
            else                             : v = -1
        elif     (2816  <= t) and (t < 3072 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 2560 ): v = 10
            elif (2560  <= s) and (s < 2816 ): v = 9
            else                             : v = -1
        elif     (3072  <= t) and (t < 3328 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 3072 ): v = 11
            else                             : v = -1
        elif     (3328  <= t) and (t < 3584 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 3072 ): v = 11
            elif (3072  <= s) and (s < 3328 ): v = 9
            else                             : v = -1
        elif     (3584  <= t) and (t < 3840 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 3072 ): v = 11
            elif (3072  <= s) and (s < 3584 ): v = 10
            else                             : v = -1
        elif     (3840  <= t) and (t < 4096 ):
            if   (0     <= s) and (s < 2048 ): v = 12
            elif (2048  <= s) and (s < 3072 ): v = 11
            elif (3072  <= s) and (s < 3584 ): v = 10
            elif (3584  <= s) and (s < 3840 ): v = 9
            else                             : v = -1
        elif     (4096  <= t) and (t < 4352 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            else                             : v = -1
        elif     (4352  <= t) and (t < 4608 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 4352 ): v = 9
            else                             : v = -1
        elif     (4608  <= t) and (t < 4864 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 4608 ): v = 10
            else                             : v = -1
        elif     (4864  <= t) and (t < 5120 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 4608 ): v = 10
            elif (4608  <= s) and (s < 4864 ): v = 9
            else                             : v = -1
        elif     (5120  <= t) and (t < 5376 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 5120 ): v = 11
            else                             : v = -1
        elif     (5376  <= t) and (t < 5632 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 5120 ): v = 11
            elif (5120  <= s) and (s < 5376 ): v = 9
            else                             : v = -1
        elif     (5632  <= t) and (t < 5888 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 5120 ): v = 11
            elif (5120  <= s) and (s < 5632 ): v = 10
            else                             : v = -1
        elif     (5888  <= t) and (t < 6144 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 5120 ): v = 11
            elif (5120  <= s) and (s < 5632 ): v = 10
            elif (5632  <= s) and (s < 5888 ): v = 9
            else                             : v = -1
        elif     (6144  <= t) and (t < 6400 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            else                             : v = -1
        elif     (6400  <= t) and (t < 6656 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 6400 ): v = 9
            else                             : v = -1
        elif     (6656  <= t) and (t < 6912 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 6656 ): v = 10
            else                             : v = -1
        elif     (6912  <= t) and (t < 7168 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 6656 ): v = 10
            elif (6656  <= s) and (s < 6912 ): v = 9
            else                             : v = -1
        elif     (7168  <= t) and (t < 7424 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 7168 ): v = 11
            else                             : v = -1
        elif     (7424  <= t) and (t < 7680 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 7168 ): v = 11
            elif (7168  <= s) and (s < 7424 ): v = 9
            else                             : v = -1
        elif     (7680  <= t) and (t < 7936 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 7168 ): v = 11
            elif (7168  <= s) and (s < 7680 ): v = 10
            else                             : v = -1
        elif     (7936  <= t) and (t < 8192 ):
            if   (0     <= s) and (s < 4096 ): v = 13
            elif (4096  <= s) and (s < 6144 ): v = 12
            elif (6144  <= s) and (s < 7168 ): v = 11
            elif (7168  <= s) and (s < 7680 ): v = 10
            elif (7680  <= s) and (s < 7936 ): v = 9
            else                             : v = -1
        elif     (8192  <= t) and (t < 8448 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            else                             : v = -1
        elif     (8448  <= t) and (t < 8704 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 8448 ): v = 9
            else                             : v = -1
        elif     (8704  <= t) and (t < 8960 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 8704 ): v = 10
            else                             : v = -1
        elif     (8960  <= t) and (t < 9216 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 8704 ): v = 10
            elif (8704  <= s) and (s < 8960 ): v = 9
            else                             : v = -1
        elif     (9216  <= t) and (t < 9472 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 9216 ): v = 11
            else                             : v = -1
        elif     (9472  <= t) and (t < 9728 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 9216 ): v = 11
            elif (9216  <= s) and (s < 9472 ): v = 9
            else                             : v = -1
        elif     (9728  <= t) and (t < 9984 ):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 9216 ): v = 11
            elif (9216  <= s) and (s < 9728 ): v = 10
            else                             : v = -1
        elif     (9984  <= t) and (t < 10240):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 9216 ): v = 11
            elif (9216  <= s) and (s < 9728 ): v = 10
            elif (9728  <= s) and (s < 9984 ): v = 9
            else                             : v = -1
        elif     (10240 <= t) and (t < 10496):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            else                             : v = -1
        elif     (10496 <= t) and (t < 10752):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 10496): v = 9
            else                             : v = -1
        elif     (10752 <= t) and (t < 11008):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 10752): v = 10
            else                             : v = -1
        elif     (11008 <= t) and (t < 11264):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 10752): v = 10
            elif (10752 <= s) and (s < 11008): v = 9
            else                             : v = -1
        elif     (11264 <= t) and (t < 11520):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 11264): v = 11
            else                             : v = -1
        elif     (11520 <= t) and (t < 11776):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 11264): v = 11
            elif (11264 <= s) and (s < 11520): v = 9
            else                             : v = -1
        elif     (11776 <= t) and (t < 12032):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 11264): v = 11
            elif (11264 <= s) and (s < 11776): v = 10
            else                             : v = -1
        elif     (12032 <= t) and (t < 12288):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 10240): v = 12
            elif (10240 <= s) and (s < 11264): v = 11
            elif (11264 <= s) and (s < 11776): v = 10
            elif (11776 <= s) and (s < 12032): v = 9
            else                             : v = -1
        elif     (12288 <= t) and (t < 12544):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            else                             : v = -1
        elif     (12544 <= t) and (t < 12800):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 12544): v = 9
            else                             : v = -1
        elif     (12800 <= t) and (t < 13056):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 12800): v = 10
            else                             : v = -1
        elif     (13056 <= t) and (t < 13312):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 12800): v = 10
            elif (12800 <= s) and (s < 13056): v = 9
            else                             : v = -1
        elif     (13312 <= t) and (t < 13568):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 13312): v = 11
            else                             : v = -1
        elif     (13568 <= t) and (t < 13824):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 13312): v = 11
            elif (13312 <= s) and (s < 13568): v = 9
            else                             : v = -1
        elif     (13824 <= t) and (t < 14080):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 13312): v = 11
            elif (13312 <= s) and (s < 13824): v = 10
            else                             : v = -1
        elif     (14080 <= t) and (t < 14336):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 13312): v = 11
            elif (13312 <= s) and (s < 13824): v = 10
            elif (13824 <= s) and (s < 14080): v = 9
            else                             : v = -1
        elif     (14336 <= t) and (t < 14592):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            else                             : v = -1
        elif     (14592 <= t) and (t < 14848):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 14592): v = 9
            else                             : v = -1
        elif     (14848 <= t) and (t < 15104):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 14848): v = 10
            else                             : v = -1
        elif     (15104 <= t) and (t < 15360):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 14848): v = 10
            elif (14848 <= s) and (s < 15104): v = 9
            else                             : v = -1
        elif     (15360 <= t) and (t < 15616):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 15360): v = 11
            else                             : v = -1
        elif     (15616 <= t) and (t < 15872):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 15360): v = 11
            elif (15360 <= s) and (s < 15616): v = 9
            else                             : v = -1
        elif     (15872 <= t) and (t < 16128):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 15360): v = 11
            elif (15360 <= s) and (s < 15872): v = 10
            else                             : v = -1
        elif     (16128 <= t) and (t < 16384):
            if   (0     <= s) and (s < 8192 ): v = 14
            elif (8192  <= s) and (s < 12288): v = 13
            elif (12288 <= s) and (s < 14336): v = 12
            elif (14336 <= s) and (s < 15360): v = 11
            elif (15360 <= s) and (s < 15872): v = 10
            elif (15872 <= s) and (s < 16128): v = 9
            else                             : v = -1
        else                                 : v = -1
    return v
