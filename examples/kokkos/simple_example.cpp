//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <Kokkos_Core.hpp>

#include <shaman.h>

template <typename ScalarType>
bool small_test(size_t n) {
    ScalarType sum = 0;
    Kokkos::parallel_reduce(
            n, KOKKOS_LAMBDA(const int i, ScalarType& lsum) { auto val = ScalarType(i+1); lsum += 1/val * 1/val; }, sum);

    std::cout <<
              "Sum from 0 to " << n-1 <<
              ", computed in parallel, is " << sum << std::endl;

    // Compare to a sequential loop.
    ScalarType seqSum = 0;
    for (int i = 0; i < n; ++i) {
        ScalarType val = i + 1;
        seqSum += 1/val * 1/val;
    }
    std::cout <<
              "Sum from 0 to " << n-1 <<
              ", computed in sequential, is " << seqSum << std::endl;
    return true;
}


int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    const int n = 1<<24;

    // Compute the sum of squares of integers from 0 to n-1, in
    // parallel, using Kokkos.  This time, use a lambda instead of a
    // functor.  The lambda takes the same arguments as the functor's
    // operator().

    small_test<double>(n);
    small_test<Sdouble>(n);

    Kokkos::finalize();

    return 0;
}
