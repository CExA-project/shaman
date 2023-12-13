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

//
// First reduction (parallel_reduce) example:
//   1. Start up Kokkos
//   2. Execute a parallel_reduce loop in the default execution space,
//      using a C++11 lambda to define the loop body
//   3. Shut down Kokkos
//
// This example only builds if C++11 is enabled.  Compare this example
// to 02_simple_reduce, which uses a functor to define the loop body
// of the parallel_reduce.
//

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    const int n = 1<<24;

    // Compute the sum of squares of integers from 0 to n-1, in
    // parallel, using Kokkos.  This time, use a lambda instead of a
    // functor.  The lambda takes the same arguments as the functor's
    // operator().
    Sdouble sum = 0;

    Kokkos::parallel_reduce(
      n, KOKKOS_LAMBDA(const int i, Sdouble& lsum) { auto val = Sdouble(i+1); lsum += 1/val * 1/val; }, sum);

    std::cout <<
            "Sum from 0 to " << n-1 <<
            ", computed in parallel, is " << sum << std::endl;

    // Compare to a sequential loop.
    Sdouble seqSum = 0;
    for (int i = 0; i < n; ++i) {
        Sdouble val = i + 1;
        seqSum += 1/val * 1/val;
    }
    std::cout <<
                 "Sum from 0 to " << n-1 <<
                 ", computed in sequential, is " << seqSum << std::endl;

    Kokkos::finalize();

    return 0;
}
