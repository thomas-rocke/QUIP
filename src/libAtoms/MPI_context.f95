! H0 XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
! H0 X
! H0 X   libAtoms+QUIP: atomistic simulation library
! H0 X
! H0 X   Portions of this code were written by
! H0 X     Albert Bartok-Partay, Silvia Cereda, Gabor Csanyi, James Kermode,
! H0 X     Ivan Solt, Wojciech Szlachta, Csilla Varnai, Steven Winfield.
! H0 X
! H0 X   Copyright 2006-2010.
! H0 X
! H0 X   These portions of the source code are released under the GNU General
! H0 X   Public License, version 2, http://www.gnu.org/copyleft/gpl.html
! H0 X
! H0 X   If you would like to license the source code under different terms,
! H0 X   please contact Gabor Csanyi, gabor@csanyi.net
! H0 X
! H0 X   Portions of this code were written by Noam Bernstein as part of
! H0 X   his employment for the U.S. Government, and are not subject
! H0 X   to copyright in the USA.
! H0 X
! H0 X
! H0 X   When using this software, please cite the following reference:
! H0 X
! H0 X   http://www.libatoms.org
! H0 X
! H0 X  Additional contributions by
! H0 X    Alessio Comisso, Chiara Gattinoni, and Gianpietro Moras
! H0 X
! H0 XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

#include "error.inc"

module MPI_context_module
use error_module
use system_module

#ifdef _MPI
#ifndef _OLDMPI
  use mpi
#endif
#endif
implicit none
#ifdef _MPI
#ifdef _OLDMPI
include 'mpif.h'
#endif
#endif

private

public :: ROOT, ROOT_
integer, parameter :: ROOT = 0
integer, parameter :: ROOT_ = ROOT  ! use if ROOT is shadowed locally

public :: MPI_context
type MPI_context
  logical :: active = .false.
  integer :: communicator = 0
  integer :: n_procs = 1
  integer :: my_proc = 0
  integer :: n_hosts = 1
  integer :: my_host = 0
  character(len=:), allocatable :: hostname
  logical :: is_cart = .false.
  integer :: my_coords(3) = 0
end type MPI_context

public :: Initialise
interface Initialise
  module procedure MPI_context_Initialise
end interface

public :: Finalise
interface Finalise
  module procedure MPI_context_Finalise
end interface

public :: is_root
interface is_root
  module procedure MPI_context_is_root
end interface

public :: Print
interface Print
  module procedure MPI_context_Print
end interface

public :: Split_context
interface Split_context
  module procedure MPI_context_Split_context
end interface

public :: free_context
interface free_context
  module procedure MPI_context_free_context
end interface

public :: bcast
interface bcast
  module procedure MPI_context_bcast_real, MPI_context_bcast_real1, MPI_context_bcast_real2
  module procedure MPI_context_bcast_c, MPI_context_bcast_c1, MPI_context_bcast_c2
  module procedure MPI_context_bcast_int, MPI_context_bcast_int1, MPI_context_bcast_int2
  module procedure MPI_context_bcast_logical, MPI_context_bcast_logical1, MPI_context_bcast_logical2
  module procedure MPI_context_bcast_char, MPI_context_bcast_char1, MPI_context_bcast_char2
end interface
public :: min
interface min
  module procedure MPI_context_min_real, MPI_context_min_int
end interface
public :: max
interface max
  module procedure MPI_context_max_real, MPI_context_max_int
end interface
public :: all
interface all
   module procedure MPI_context_all
endinterface
public :: any
interface any
   module procedure MPI_context_any
endinterface
public :: sum
interface sum
  module procedure MPI_context_sum_int
  module procedure MPI_context_sum_real
  module procedure MPI_context_sum_complex
end interface
public :: sum_in_place
interface sum_in_place
  module procedure MPI_context_sum_in_place_int0
  module procedure MPI_context_sum_in_place_int1
  module procedure MPI_context_sum_in_place_real0
  module procedure MPI_context_sum_in_place_real1
  module procedure MPI_context_sum_in_place_real2
  module procedure MPI_context_sum_in_place_real3
  module procedure MPI_context_sum_in_place_complex1
  module procedure MPI_context_sum_in_place_complex2
end interface

public :: gather
interface gather
  module procedure MPI_context_gather_char0
end interface gather

public :: gatherv
interface gatherv
  module procedure MPI_context_gatherv_int1
  module procedure MPI_context_gatherv_real2
end interface gatherv

public :: allgatherv
interface allgatherv
  module procedure MPI_context_allgatherv_real2
end interface allgatherv

public :: scatter
interface scatter
  module procedure MPI_context_scatter_int0
end interface scatter

public :: scatterv
interface scatterv
  module procedure MPI_context_scatterv_int1
  module procedure MPI_context_scatterv_real1
  module procedure MPI_context_scatterv_real2
end interface scatterv

public :: mpi_print

public :: barrier
interface barrier
  module procedure MPI_context_barrier
end interface barrier

public :: cart_shift
interface cart_shift
   module procedure MPI_context_cart_shift
end interface cart_shift

public :: sendrecv
interface sendrecv
   module procedure MPI_context_sendrecv_c1a
   module procedure MPI_context_sendrecv_r, MPI_context_sendrecv_ra
end interface sendrecv

public :: push_MPI_error

contains

subroutine MPI_context_Initialise(this, communicator, context, dims, periods, error)
  type(MPI_context), intent(inout) :: this
  integer, intent(in), optional :: communicator
  type(MPI_context), intent(in), optional :: context
  integer, intent(in), optional :: dims(3)
  logical, intent(in), optional :: periods(3)
  integer, intent(out), optional :: error

#ifdef _MPI
  integer :: err
  logical :: is_initialized
  integer :: comm
  logical :: my_periods(3)
#endif

  INIT_ERROR(error)

  if (present(communicator) .and. present(context)) then
     RAISE_ERROR("Please specify either *communicator* or *context* upon call to MPI_context_Initialise.", error)
  endif

  call Finalise(this)

  this%active = .false.
  this%is_cart = .false.

#ifndef _MPI
  allocate(character(1) :: this%hostname)
  this%hostname = ' '
#endif

#ifdef _MPI
  if (present(communicator) .or. present(context)) then

     if (present(communicator)) then
        comm = communicator
     else if (present(context)) then
        comm = context%communicator
     endif

  else

     comm = MPI_COMM_WORLD
     call mpi_initialized(is_initialized, err)
     PASS_MPI_ERROR(err, error)
     if (.not. is_initialized) then
        call mpi_init(err)
        PASS_MPI_ERROR(err, error)
     endif

  endif
  this%communicator = comm

  if (present(dims)) then
     my_periods = .true.
     if (present(periods)) then
        my_periods = periods
     endif

     this%is_cart = .true.
     call mpi_cart_create(comm, 3, dims, my_periods, .true., &
          this%communicator, err)
     PASS_MPI_ERROR(err, error)
  endif

  call mpi_comm_set_errhandler(this%communicator, MPI_ERRORS_RETURN, err)
  PASS_MPI_ERROR(err, error)

  call mpi_comm_size(this%communicator, this%n_procs, err)
  PASS_MPI_ERROR(err, error)
  call mpi_comm_rank(this%communicator, this%my_proc, err)
  PASS_MPI_ERROR(err, error)
  this%active = .true.

  if (this%is_cart) then
     call mpi_cart_coords(this%communicator, this%my_proc, 3, &
          this%my_coords, err)
     PASS_MPI_ERROR(err, error)

     call print("MPI_context_Initialise : Cart created, coords = " // this%my_coords, PRINT_VERBOSE)
  endif

  call MPI_context_init_hostname(this, err)
  PASS_MPI_ERROR(err, error)
  call MPI_context_scatter_hosts(this, err)
  PASS_MPI_ERROR(err, error)
#endif
end subroutine MPI_context_Initialise

subroutine MPI_context_Finalise(this, end_of_program, error)
  type(MPI_context), intent(inout) :: this
  logical, optional, intent(in) :: end_of_program
  integer, intent(out), optional :: error

#ifdef _MPI
  integer :: err
  logical :: is_initialized
#endif

  INIT_ERROR(error)

  if (.not. this%active) return

#ifdef _MPI
  if (present(end_of_program)) then
    if (end_of_program) then
      call mpi_initialized(is_initialized, err)
      PASS_MPI_ERROR(err, error)
      if (.not. is_initialized) then
        call mpi_finalize(err)
        PASS_MPI_ERROR(err, error)
      endif
    endif
  endif
#endif
end subroutine MPI_context_Finalise

!% Set hostname with max length among all MPI processes
subroutine MPI_context_init_hostname(this, error)
  type(MPI_context), intent(inout) :: this
  integer, intent(out), optional :: error

#ifdef _MPI
  integer :: err, my_len, max_len
  character(MPI_MAX_PROCESSOR_NAME) :: hostname
#endif

  INIT_ERROR(error)

  if (.not. this%active) return

#ifdef _MPI
  call mpi_get_processor_name(hostname, my_len, err)
  PASS_MPI_ERROR(err, error)

  max_len = max(this, my_len, err)
  PASS_MPI_ERROR(err, error)

  if (allocated(this%hostname)) deallocate(this%hostname)
  allocate(character(my_len) :: this%hostname)
  this%hostname = hostname(1:max_len)
#endif
end subroutine MPI_context_init_hostname

!% Set host index and total for each MPI process
subroutine MPI_context_scatter_hosts(this, error)
  type(MPI_context), intent(inout) :: this
  integer, intent(out), optional :: error

  integer :: err, i
  character(:), allocatable :: hostnames(:)
  integer, allocatable :: refs(:), hosts(:)

  INIT_ERROR(error)

  if (.not. this%active) return

#ifdef _MPI
  if (.not. allocated(this%hostname)) then
    call mpi_context_init_hostname(this)
  end if

  if (is_root(this)) then
    allocate(character(len(this%hostname)) :: hostnames(0:this%n_procs-1))
  else
    allocate(character(1) :: hostnames(0:0))
  end if

  call gather(this, this%hostname, hostnames, error=err)
  PASS_MPI_ERROR(err, error)

  if (is_root(this)) then
    call get_uniqs_refs(hostnames, hosts, refs, lbound(hostnames, 1))
    this%n_hosts = size(hosts)
  else
    allocate(hosts(1), source=0)
    allocate(refs(1), source=0)
  end if

  call scatter(this, refs, this%my_host, error=err)
  PASS_MPI_ERROR(err, error)
  call bcast(this, this%n_hosts, error=err)
  PASS_MPI_ERROR(err, error)

  if (is_root(this)) then
    call print("MPI hostnames :: "//join(hostnames(hosts), " "))
    call print("MPI host refs :: "//refs)
  end if
  call print("MPI my_host  : "//this%my_host)
  call print("MPI hostname : "//this%hostname)
#endif
end subroutine MPI_context_scatter_hosts

function MPI_context_is_root(this, root) result(res)
  type(MPI_context), intent(in) :: this
  integer, intent(in), optional :: root
  logical :: res
  if (this%active) then
    res = (this%my_proc == optional_default(ROOT_, root))
  else
    res = .true.
  end if
end function MPI_context_is_root

subroutine MPI_context_free_context(this, error)
  type(MPI_context), intent(inout) :: this
  integer, intent(out), optional :: error

#ifdef _MPI
  integer err
#endif

  INIT_ERROR(error)

  if (.not. this%active) return

#ifdef _MPI
  call mpi_comm_free(this%communicator, err)
  PASS_MPI_ERROR(err, error)
  this%active = .false.
#endif
end subroutine MPI_context_free_context

subroutine MPI_context_Split_context(this, split_index, new_context, error)
  type(MPI_context), intent(in) :: this
  integer, intent(in) :: split_index
  type(MPI_context), intent(out) :: new_context
  integer, intent(out), optional :: error

#ifdef _MPI
  integer err
#endif
  integer new_comm

  INIT_ERROR(error)

  if (.not. this%active) then
    new_context = this
    return
  endif

  new_comm = this%communicator

#ifdef _MPI
  call mpi_comm_split(this%communicator, split_index, this%my_proc, new_comm, err)
  PASS_MPI_ERROR(err, error)
#endif

  call Initialise(new_context, new_comm)

end subroutine MPI_context_Split_context

function MPI_context_min_real(this, v, error)
  type(MPI_context), intent(in) :: this
  real(dp), intent(in) :: v
  integer, intent(out), optional :: error
  real(dp) :: MPI_context_min_real

#ifdef _MPI
  integer err
#endif

  INIT_ERROR(error)

  if (.not. this%active) then
    MPI_context_min_real = v
    return
  endif

#ifdef _MPI
  call MPI_allreduce(v, MPI_context_min_real, 1, MPI_DOUBLE_PRECISION, MPI_MIN, this%communicator, err)
  PASS_MPI_ERROR(err, error)
#else
  MPI_context_min_real = v
#endif
end function MPI_context_min_real

function MPI_context_min_int(this, v, error)
  type(MPI_context), intent(in) :: this
  integer, intent(in) :: v
  integer, intent(out), optional :: error
  integer :: MPI_context_min_int

#ifdef _MPI
  integer err
#endif

  INIT_ERROR(error)

  if (.not. this%active) then
    MPI_context_min_int = v
    return
  endif

#ifdef _MPI
  call MPI_allreduce(v, MPI_context_min_int, 1, MPI_INTEGER, MPI_MIN, this%communicator, err)
  PASS_MPI_ERROR(err, error)
#else
  MPI_context_min_int = v
#endif
end function MPI_context_min_int

function MPI_context_max_real(this, v, error)
  type(MPI_context), intent(in) :: this
  real(dp), intent(in) :: v
  integer, intent(out), optional :: error
  real(dp) :: MPI_context_max_real

#ifdef _MPI
  integer err
#endif

  INIT_ERROR(error)

  if (.not. this%active) then
    MPI_context_max_real = v
    return
  endif

#ifdef _MPI
  call MPI_allreduce(v, MPI_context_max_real, 1, MPI_DOUBLE_PRECISION, MPI_MAX, this%communicator, err)
  PASS_MPI_ERROR(err, error)
#else
  MPI_context_max_real = v
#endif
end function MPI_context_max_real

function MPI_context_max_int(this, v, error)
  type(MPI_context), intent(in) :: this
  integer, intent(in) :: v
  integer, intent(out), optional :: error
  integer :: MPI_context_max_int

#ifdef _MPI
  integer err
#endif

  INIT_ERROR(error)

  if (.not. this%active) then
    MPI_context_max_int = v
    return
  endif

#ifdef _MPI
  call MPI_allreduce(v, MPI_context_max_int, 1, MPI_INTEGER, MPI_MAX, this%communicator, err)
  PASS_MPI_ERROR(err, error)
#else
  MPI_context_max_int = v
#endif
end function MPI_context_max_int

function MPI_context_all(this, v, error)
  type(MPI_context), intent(in) :: this
  logical, intent(in) :: v
  integer, intent(out), optional :: error
  logical :: MPI_context_all

#ifdef _MPI
  integer err
#endif

  INIT_ERROR(error)

  if (.not. this%active) then
    MPI_context_all = v
    return
  endif

#ifdef _MPI
  call MPI_allreduce(v, MPI_context_all, 1, MPI_LOGICAL, MPI_LAND, this%communicator, err)
  PASS_MPI_ERROR(err, error)
#else
  MPI_context_all = v
#endif
end function MPI_context_all

function MPI_context_any(this, v, error)
  type(MPI_context), intent(in) :: this
  logical, intent(in) :: v
  integer, intent(out), optional :: error
  logical :: MPI_context_any

#ifdef _MPI
  integer err
#endif

  INIT_ERROR(error)

  if (.not. this%active) then
    MPI_context_any = v
    return
  endif

#ifdef _MPI
  call MPI_allreduce(v, MPI_context_any, 1, MPI_LOGICAL, MPI_LOR, this%communicator, err)
  PASS_MPI_ERROR(err, error)
#else
  MPI_context_any = v
#endif
end function MPI_context_any

function MPI_context_sum_int(this, v, error)
  type(MPI_context), intent(in) :: this
  integer, intent(in) :: v
  integer, intent(out), optional :: error
  integer :: MPI_context_sum_int

#ifdef _MPI
  integer err
#endif

  INIT_ERROR(error)

  if (.not. this%active) then
    MPI_context_sum_int = v
    return
  endif

#ifdef _MPI
  call MPI_allreduce(v, MPI_context_sum_int, 1, MPI_INTEGER, MPI_SUM, this%communicator, err)
  PASS_MPI_ERROR(err, error)
#else
  MPI_context_sum_int = v
#endif
end function  MPI_context_sum_int

function MPI_context_sum_real(this, v, error)
  type(MPI_context), intent(in) :: this
  real(dp), intent(in) :: v
  integer, intent(out), optional :: error
  real(dp) :: MPI_context_sum_real

#ifdef _MPI
  integer err
#endif

  INIT_ERROR(error)

  if (.not. this%active) then
    MPI_context_sum_real = v
    return
  endif

#ifdef _MPI
  call MPI_allreduce(v, MPI_context_sum_real, 1, MPI_DOUBLE_PRECISION, MPI_SUM, this%communicator, err)
  PASS_MPI_ERROR(err, error)
#else
  MPI_context_sum_real = v
#endif
end function  MPI_context_sum_real

function MPI_context_sum_complex(this, v, error)
  type(MPI_context), intent(in) :: this
  complex(dp), intent(in) :: v
  integer, intent(out), optional :: error
  complex(dp) :: MPI_context_sum_complex

#ifdef _MPI
  integer err
#endif

  INIT_ERROR(error)

  if (.not. this%active) then
    MPI_context_sum_complex = v
    return
  endif

#ifdef _MPI
  call MPI_allreduce(v, MPI_context_sum_complex, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, this%communicator, err)
  PASS_MPI_ERROR(err, error)
#else
  MPI_context_sum_complex = v
#endif
end function MPI_context_sum_complex

subroutine MPI_context_sum_in_place_real2(this, v, error)
  type(MPI_context), intent(in) :: this
  real(dp), intent(inout) :: v(:,:)
  integer, intent(out), optional :: error

#ifdef MPI_1
  real(dp), allocatable :: v_sum(:,:)
#endif
#ifdef _MPI
  integer err
#endif

  INIT_ERROR(error)

  if (.not. this%active) return

#ifdef _MPI
#ifdef MPI_1
  allocate(v_sum(size(v,1),size(v,2)))
  call MPI_allreduce(v, v_sum, size(v), MPI_DOUBLE_PRECISION, MPI_SUM, this%communicator, err)
  v = v_sum
#else
  call MPI_allreduce(MPI_IN_PLACE, v, size(v), MPI_DOUBLE_PRECISION, MPI_SUM, this%communicator, err)
#endif
  PASS_MPI_ERROR(err, error)
#endif
end subroutine MPI_context_sum_in_place_real2

subroutine MPI_context_sum_in_place_real3(this, v, error)
  type(MPI_context), intent(in) :: this
  real(dp), intent(inout) :: v(:,:,:)
  integer, intent(out), optional :: error

#ifdef MPI_1
  real(dp), allocatable :: v_sum(:,:,:)
#endif
#ifdef _MPI
  integer err
#endif

  INIT_ERROR(error)

  if (.not. this%active) return

#ifdef _MPI
#ifdef MPI_1
  allocate(v_sum(size(v,1),size(v,2),size(v,3)))
  call MPI_allreduce(v, v_sum, size(v), MPI_DOUBLE_PRECISION, MPI_SUM, this%communicator, err)
  v = v_sum
#else
  call MPI_allreduce(MPI_IN_PLACE, v, size(v), MPI_DOUBLE_PRECISION, MPI_SUM, this%communicator, err)
#endif
  PASS_MPI_ERROR(err, error)
#endif
end subroutine  MPI_context_sum_in_place_real3

subroutine MPI_context_sum_in_place_complex2(this, v, error)
  type(MPI_context), intent(in) :: this
  complex(dp), intent(inout) :: v(:,:)
  integer, intent(out), optional :: error

#ifdef MPI_1
  complex(dp), allocatable :: v_sum(:,:)
#endif
#ifdef _MPI
  integer err
#endif

  INIT_ERROR(error)

  if (.not. this%active) return

#ifdef _MPI
#ifdef MPI_1
  allocate(v_sum(size(v,1),size(v,2)))
  call MPI_allreduce(v, v_sum, size(v), MPI_DOUBLE_COMPLEX, MPI_SUM, this%communicator, err)
  v = v_sum
#else
  call MPI_allreduce(MPI_IN_PLACE, v, size(v), MPI_DOUBLE_COMPLEX, MPI_SUM, this%communicator, err)
#endif
  PASS_MPI_ERROR(err, error)
#endif
end subroutine  MPI_context_sum_in_place_complex2

subroutine MPI_context_sum_in_place_int0(this, v, error)
  type(MPI_context), intent(in) :: this
  integer, intent(inout) :: v
  integer, intent(out), optional :: error

#ifdef MPI_1
  integer :: v_sum
#endif
#ifdef _MPI
  integer :: err
#endif

  INIT_ERROR(error)

  if (.not. this%active) return

#ifdef _MPI
#ifdef MPI_1
  call MPI_allreduce(v, v_sum, 1, MPI_INTEGER, MPI_SUM, this%communicator, err)
  v = v_sum
#else
  call MPI_allreduce(MPI_IN_PLACE, v, 1, MPI_INTEGER, MPI_SUM, this%communicator, err)
#endif
  PASS_MPI_ERROR(err, error)
#endif
end subroutine MPI_context_sum_in_place_int0

subroutine MPI_context_sum_in_place_int1(this, v, error)
  type(MPI_context), intent(in) :: this
  integer, intent(inout) :: v(:)
  integer, intent(out), optional :: error

#ifdef MPI_1
  integer, allocatable :: v_sum(:)
#endif
#ifdef _MPI
  integer err
#endif

  INIT_ERROR(error)

  if (.not. this%active) return

#ifdef _MPI
#ifdef MPI_1
  allocate(v_sum(size(v,1)))
  call MPI_allreduce(v, v_sum, size(v), MPI_INTEGER, MPI_SUM, this%communicator, err)
  v = v_sum
#else
  call MPI_allreduce(MPI_IN_PLACE, v, size(v), MPI_INTEGER, MPI_SUM, this%communicator, err)
#endif
  PASS_MPI_ERROR(err, error)
#endif
end subroutine MPI_context_sum_in_place_int1

subroutine MPI_context_sum_in_place_real0(this, v, error)
  type(MPI_context), intent(in) :: this
  real(dp), intent(inout) :: v
  integer, intent(out), optional :: error

#ifdef MPI_1
  real(dp) :: v_sum
#endif
#ifdef _MPI
  integer err
#endif

  INIT_ERROR(error)

  if (.not. this%active) return

#ifdef _MPI
#ifdef MPI_1
  call MPI_allreduce(v, v_sum, 1, MPI_DOUBLE_PRECISION, MPI_SUM, this%communicator, err)
  v = v_sum
#else
  call MPI_allreduce(MPI_IN_PLACE, v, 1, MPI_DOUBLE_PRECISION, MPI_SUM, this%communicator, err)
#endif
  PASS_MPI_ERROR(err, error)
#endif
end subroutine MPI_context_sum_in_place_real0

subroutine MPI_context_sum_in_place_real1(this, v, error)
  type(MPI_context), intent(in) :: this
  real(dp), intent(inout) :: v(:)
  integer, intent(out), optional :: error

#ifdef MPI_1
  real(dp), allocatable :: v_sum(:)
#endif
#ifdef _MPI
  integer err
#endif

  INIT_ERROR(error)

  if (.not. this%active) return

#ifdef _MPI
#ifdef MPI_1
  allocate(v_sum(size(v,1)))
  call MPI_allreduce(v, v_sum, size(v), MPI_DOUBLE_PRECISION, MPI_SUM, this%communicator, err)
  v = v_sum
#else
  call MPI_allreduce(MPI_IN_PLACE, v, size(v), MPI_DOUBLE_PRECISION, MPI_SUM, this%communicator, err)
#endif
  PASS_MPI_ERROR(err, error)
#endif
end subroutine MPI_context_sum_in_place_real1

subroutine MPI_context_sum_in_place_complex1(this, v, error)
  type(MPI_context), intent(in) :: this
  complex(dp), intent(inout) :: v(:)
  integer, intent(out), optional :: error

#ifdef MPI_1
  complex(dp), allocatable :: v_sum(:)
#endif
#ifdef _MPI
  integer err
#endif

  INIT_ERROR(error)

  if (.not. this%active) return

#ifdef _MPI
#ifdef MPI_1
  allocate(v_sum(size(v,1)))
  call MPI_allreduce(v, v_sum, size(v), MPI_DOUBLE_COMPLEX, MPI_SUM, this%communicator, err)
  v = v_sum
#else
  call MPI_allreduce(MPI_IN_PLACE, v, size(v), MPI_DOUBLE_COMPLEX, MPI_SUM, this%communicator, err)
#endif
  PASS_MPI_ERROR(err, error)
#endif
end subroutine  MPI_context_sum_in_place_complex1

subroutine MPI_context_bcast_int(this, v, root, error)
  type(MPI_context), intent(in) :: this
  integer, intent(inout) :: v
  integer, intent(in), optional :: root
  integer, intent(out), optional :: error

#ifdef _MPI
  integer err
  integer my_root
#endif

  INIT_ERROR(error)

  if (.not. this%active) return

#ifdef _MPI
  my_root = optional_default(ROOT_, root)
  call MPI_Bcast(v, 1, MPI_INTEGER, my_root, this%communicator, err)
  PASS_MPI_ERROR(err, error)
#endif
end subroutine MPI_context_bcast_int

subroutine MPI_context_bcast_int1(this, v, root, error)
  type(MPI_context), intent(in) :: this
  integer, intent(inout) :: v(:)
  integer, intent(in), optional :: root
  integer, intent(out), optional :: error

#ifdef _MPI
  integer err
  integer my_root
#endif

  INIT_ERROR(error)

  if (.not. this%active) return

#ifdef _MPI
  my_root = optional_default(ROOT_, root)
  call MPI_Bcast(v, size(v), MPI_INTEGER, my_root, this%communicator, err)
  PASS_MPI_ERROR(err, error)
#endif
end subroutine MPI_context_bcast_int1

subroutine MPI_context_bcast_int2(this, v, root, error)
  type(MPI_context), intent(in) :: this
  integer, intent(inout) :: v(:,:)
  integer, intent(in), optional :: root
  integer, intent(out), optional :: error

#ifdef _MPI
  integer err
  integer my_root
#endif

  INIT_ERROR(error)

  if (.not. this%active) return

#ifdef _MPI
  my_root = optional_default(ROOT_, root)
  call MPI_Bcast(v, size(v), MPI_INTEGER, my_root, this%communicator, err)
  PASS_MPI_ERROR(err, error)
#endif
end subroutine MPI_context_bcast_int2


subroutine MPI_context_bcast_logical(this, v, root, error)
  type(MPI_context), intent(in) :: this
  logical, intent(inout) :: v
  integer, intent(in), optional :: root
  integer, intent(out), optional :: error

#ifdef _MPI
  integer err
  integer my_root
#endif

  INIT_ERROR(error)

  if (.not. this%active) return

#ifdef _MPI
  my_root = optional_default(ROOT_, root)
  call MPI_Bcast(v, 1, MPI_LOGICAL, my_root, this%communicator, err)
  PASS_MPI_ERROR(err, error)
#endif
end subroutine MPI_context_bcast_logical

subroutine MPI_context_bcast_logical1(this, v, root, error)
  type(MPI_context), intent(in) :: this
  logical, intent(inout) :: v(:)
  integer, intent(in), optional :: root
  integer, intent(out), optional :: error

#ifdef _MPI
  integer err
  integer my_root
#endif

  INIT_ERROR(error)

  if (.not. this%active) return

#ifdef _MPI
  my_root = optional_default(ROOT_, root)
  call MPI_Bcast(v, size(v), MPI_LOGICAL, my_root, this%communicator, err)
  PASS_MPI_ERROR(err, error)
#endif
end subroutine MPI_context_bcast_logical1

subroutine MPI_context_bcast_logical2(this, v, root, error)
  type(MPI_context), intent(in) :: this
  logical, intent(inout) :: v(:,:)
  integer, intent(in), optional :: root
  integer, intent(out), optional :: error

#ifdef _MPI
  integer err
  integer my_root
#endif

  INIT_ERROR(error)

  if (.not. this%active) return

#ifdef _MPI
  my_root = optional_default(ROOT_, root)
  call MPI_Bcast(v, size(v), MPI_LOGICAL, my_root, this%communicator, err)
  PASS_MPI_ERROR(err, error)
#endif
end subroutine MPI_context_bcast_logical2


subroutine MPI_context_bcast_c(this, v, root, error)
  type(MPI_context), intent(in) :: this
  complex(dp), intent(inout) :: v
  integer, intent(in), optional :: root
  integer, intent(out), optional :: error

#ifdef _MPI
  integer err
  integer my_root
#endif

  INIT_ERROR(error)

  if (.not. this%active) return

#ifdef _MPI
  my_root = optional_default(ROOT_, root)
  call MPI_Bcast(v, 1, MPI_DOUBLE_COMPLEX, my_root, this%communicator, err)
  PASS_MPI_ERROR(err, error)
#endif
end subroutine MPI_context_bcast_c

subroutine MPI_context_bcast_c1(this, v, root, error)
  type(MPI_context), intent(in) :: this
  complex(dp), intent(inout) :: v(:)
  integer, intent(in), optional :: root
  integer, intent(out), optional :: error

#ifdef _MPI
  integer err
  integer my_root
#endif

  INIT_ERROR(error)

  if (.not. this%active) return

#ifdef _MPI
  my_root = optional_default(ROOT_, root)
  call MPI_Bcast(v, size(v), MPI_DOUBLE_COMPLEX, my_root, this%communicator, err)
  PASS_MPI_ERROR(err, error)
#endif
end subroutine MPI_context_bcast_c1

subroutine MPI_context_bcast_c2(this, v, root, error)
  type(MPI_context), intent(in) :: this
  complex(dp), intent(inout) :: v(:,:)
  integer, intent(in), optional :: root
  integer, intent(out), optional :: error

#ifdef _MPI
  integer err
  integer my_root
#endif

  INIT_ERROR(error)

  if (.not. this%active) return

#ifdef _MPI
  my_root = optional_default(ROOT_, root)
  call MPI_Bcast(v, size(v), MPI_DOUBLE_COMPLEX, my_root, this%communicator, err)
  PASS_MPI_ERROR(err, error)
#endif
end subroutine MPI_context_bcast_c2

subroutine MPI_context_bcast_real(this, v, root, error)
  type(MPI_context), intent(in) :: this
  real(dp), intent(inout) :: v
  integer, intent(in), optional :: root
  integer, intent(out), optional :: error

#ifdef _MPI
  integer err
  integer my_root
#endif

  INIT_ERROR(error)

  if (.not. this%active) return

#ifdef _MPI
  my_root = optional_default(ROOT_, root)
  call MPI_Bcast(v, 1, MPI_DOUBLE_PRECISION, my_root, this%communicator, err)
  PASS_MPI_ERROR(err, error)
#endif
end subroutine MPI_context_bcast_real

subroutine MPI_context_bcast_real1(this, v, root, error)
  type(MPI_context), intent(in) :: this
  real(dp), intent(inout) :: v(:)
  integer, intent(in), optional :: root
  integer, intent(out), optional :: error

#ifdef _MPI
  integer err
  integer my_root
#endif

  INIT_ERROR(error)

  if (.not. this%active) return

#ifdef _MPI
  my_root = optional_default(ROOT_, root)
  call MPI_Bcast(v, size(v), MPI_DOUBLE_PRECISION, my_root, this%communicator, err)
  PASS_MPI_ERROR(err, error)
#endif
end subroutine MPI_context_bcast_real1

subroutine MPI_context_bcast_real2(this, v, root, error)
  type(MPI_context), intent(in) :: this
  real(dp), intent(inout) :: v(:,:)
  integer, intent(in), optional :: root
  integer, intent(out), optional :: error

#ifdef _MPI
  integer err
  integer my_root
#endif

  INIT_ERROR(error)

  if (.not. this%active) return

#ifdef _MPI
  my_root = optional_default(ROOT_, root)
  call MPI_Bcast(v, size(v), MPI_DOUBLE_PRECISION, my_root, this%communicator, err)
  PASS_MPI_ERROR(err, error)
#endif
end subroutine MPI_context_bcast_real2

subroutine MPI_context_bcast_char(this, v, root, error)
  type(MPI_context), intent(in) :: this
  character(*), intent(inout) :: v
  integer, intent(in), optional :: root
  integer, intent(out), optional :: error

#ifdef _MPI
  integer err
  integer my_root
#endif

  INIT_ERROR(error)

  if (.not. this%active) return

#ifdef _MPI
  my_root = optional_default(ROOT_, root)
  call MPI_Bcast(v, len(v), MPI_CHARACTER, my_root, this%communicator, err)
  PASS_MPI_ERROR(err, error)
#endif
end subroutine MPI_context_bcast_char

subroutine MPI_context_bcast_char1(this, v, root, error)
  type(MPI_context), intent(in) :: this
  character(*), intent(inout) :: v(:)
  integer, intent(in), optional :: root
  integer, intent(out), optional :: error

#ifdef _MPI
  integer err
  integer my_root
#endif

  INIT_ERROR(error)

  if (.not. this%active) return

#ifdef _MPI
  my_root = optional_default(ROOT_, root)
  call MPI_Bcast(v, len(v(1))*size(v), MPI_CHARACTER, my_root, this%communicator, err)
  PASS_MPI_ERROR(err, error)
#endif
end subroutine MPI_context_bcast_char1

subroutine MPI_context_bcast_char2(this, v, root, error)
  type(MPI_context), intent(in) :: this
  character(*), intent(inout) :: v(:,:)
  integer, intent(in), optional :: root
  integer, intent(out), optional :: error

#ifdef _MPI
  integer err
  integer my_root
#endif

  INIT_ERROR(error)

  if (.not. this%active) return

#ifdef _MPI
  my_root = optional_default(ROOT_, root)
  call MPI_Bcast(v, len(v(1,1))*size(v), MPI_CHARACTER, my_root, this%communicator, err)
  PASS_MPI_ERROR(err, error)
#endif
end subroutine MPI_context_bcast_char2

subroutine MPI_Context_Print(this, file)
  type(MPI_context), intent(in) :: this
  type(Inoutput), intent(inout), optional :: file

  call print("MPI_Context : active " // this%active, file=file)
  if (this%active) then
    call print("communicator " // this%communicator, file=file)
    call print("n_procs " // this%n_procs // " my_proc " // this%my_proc, file=file)
  endif
end subroutine MPI_Context_Print

subroutine MPI_Print(this, lines, file)
  type(MPI_context), intent(in) :: this
  character(len=*), intent(in) :: lines(:)
  type(Inoutput), intent(inout), optional :: file

  integer i

  if (.not. this%active) then
    do i=1, size(lines)
      call Print(trim(lines(i)), file=file)
    end do
#ifdef _MPI
  else
    call parallel_print(lines, this%communicator, file=file)
#endif
  endif

end subroutine MPI_Print

subroutine MPI_context_gather_char0(this, v_in, v_out, root, error)
  type(MPI_context), intent(in) :: this
  character(*), intent(in) :: v_in
  character(*), intent(out) :: v_out(:)
  integer, intent(in), optional :: root
  integer, intent(out), optional :: error

  integer :: my_root, err, my_len

  INIT_ERROR(error)

  if (.not. this%active) then
    if (size(v_out) < 1) then
      RAISE_ERROR("MPI_context_gather_char0 size(v_out) < 1: " // size(v_out), error)
    end if

    if (len(v_in) /= len(v_out(1))) then
      RAISE_ERROR("MPI_context_gather_char0 lengths differ: len(v_in) " // len(v_in) // " len(v_out(1)) " // len(v_out), error)
    end if

    v_out(1) = v_in
    return
  end if

#ifdef _MPI
  my_root = optional_default(ROOT_, root)

  if (is_root(this)) then
    if (size(v_out) /= this%n_procs) then
      RAISE_ERROR("MPI_context_gather_char0 size(v_out) < %n_procs: " // size(v_out) // " " // this%n_procs, error)
    end if
    if (len(v_in) /= len(v_out(1))) then
      RAISE_ERROR("MPI_context_gather_char0 lengths differ: len(v_in) " // len(v_in) // " len(v_out(1)) " // len(v_out), error)
    end if
  end if

  call mpi_gather(v_in, len(v_in), MPI_CHARACTER, v_out, len(v_in), MPI_CHARACTER, my_root, this%communicator, err)
  PASS_MPI_ERROR(err, error)
#endif
end subroutine MPI_context_gather_char0

subroutine MPI_context_gatherv_int1(this, v_in, v_out, counts, root, error)
  type(MPI_context), intent(in) :: this
  integer, intent(in) :: v_in(:)
  integer, intent(out) :: v_out(:)
  integer, intent(out), allocatable, optional :: counts(:)
  integer, intent(in), optional :: root
  integer, intent(out), optional :: error

  integer :: my_root, err, my_count
  integer, allocatable :: displs(:), my_counts(:)

  INIT_ERROR(error)

  if (.not. this%active) then
    if (any(shape(v_in) /= shape(v_out))) then
      RAISE_ERROR("MPI_context_gatherv_int1 (no MPI) shape mismatch v_in " // shape(v_in) // " v_out " // shape(v_out), error)
    endif
    v_out = v_in
    return
  endif

#ifdef _MPI
  my_root = optional_default(ROOT_, root)
  if (is_root(this)) then
    allocate(my_counts(this%n_procs))
  else
    allocate(my_counts(1), source=0)
  end if

  my_count = size(v_in)
  call mpi_gather(my_count, 1, MPI_INTEGER, my_counts, 1, MPI_INTEGER, my_root, this%communicator, err)
  PASS_MPI_ERROR(err, error)

  if (sum(my_counts) > size(v_out)) then
    RAISE_ERROR("MPI_context_gatherv_int1 not enough space sum(my_counts) " // sum(my_counts) // " size(v_out) " // size(v_out), error)
  endif

  if (is_root(this)) then
    call get_displs(my_counts, displs)
  else
    allocate(displs(1), source=0)
  end if

  call MPI_gatherv(v_in, my_count, MPI_INTEGER, v_out, my_counts, displs, MPI_INTEGER, my_root, this%communicator, err)
  PASS_MPI_ERROR(err, error)

  if (present(counts)) then
    allocate(counts, source=my_counts)
  end if
#endif

end subroutine MPI_context_gatherv_int1

subroutine MPI_context_gatherv_real2(this, v_in, v_out, counts, root, error)
  type(MPI_context), intent(in) :: this
  real(dp), intent(in) :: v_in(:,:)
  real(dp), intent(out) :: v_out(:,:)
  integer, intent(out), allocatable, optional :: counts(:)
  integer, intent(in), optional :: root
  integer, intent(out), optional :: error

  integer :: my_root, err, my_count
  integer, allocatable :: displs(:), my_counts(:)

  INIT_ERROR(error)

  if (.not. this%active) then
    if (any(shape(v_in) /= shape(v_out))) then
      RAISE_ERROR("MPI_context_gatherv_real2 (no MPI) shape mismatch v_in " // shape(v_in) // " v_out " // shape(v_out), error)
    endif
    v_out = v_in
    return
  endif

#ifdef _MPI
  my_root = optional_default(ROOT_, root)
  if (is_root(this)) then
    allocate(my_counts(this%n_procs))
  else
    allocate(my_counts(1), source=0)
  end if

  my_count = size(v_in)
  call mpi_gather(my_count, 1, MPI_INTEGER, my_counts, 1, MPI_INTEGER, my_root, this%communicator, err)
  PASS_MPI_ERROR(err, error)

  if (sum(my_counts) > size(v_out)) then
    RAISE_ERROR("MPI_context_gatherv_real2 not enough space sum(my_counts) " // sum(my_counts) // " size(v_out) " // size(v_out), error)
  endif

  if (is_root(this)) then
    call get_displs(my_counts, displs)
  else
    allocate(displs(1), source=0)
  end if

  call MPI_gatherv(v_in, my_count, MPI_DOUBLE_PRECISION, v_out, my_counts, displs, MPI_DOUBLE_PRECISION, my_root, this%communicator, err)
  PASS_MPI_ERROR(err, error)

  if (present(counts)) then
    allocate(counts, source=my_counts)
  end if
#endif

end subroutine MPI_context_gatherv_real2

subroutine MPI_context_allgatherv_real2(this, v_in, v_out, error)
  type(MPI_context), intent(in) :: this
  real(dp), intent(in) :: v_in(:,:)
  real(dp), intent(out) :: v_out(:,:)
  integer, intent(out), optional :: error

  integer err
  integer my_count
  integer, allocatable :: displs(:), counts(:)

  INIT_ERROR(error)

  if (.not. this%active) then
    if (size(v_in,1) /= size(v_out,1) .or. &
        size(v_in,2) /= size(v_out,2)) then
      RAISE_ERROR("MPI_context_allgatherv_real (no MPI) size mismatch v_in " // shape(v_in) // " v_out " // shape(v_out), error)
    endif
    v_out = v_in
    return
  endif

#ifdef _MPI

  my_count = size(v_in)
  allocate(counts(this%n_procs))
  call mpi_allgather(my_count, 1, MPI_INTEGER, counts, 1, MPI_INTEGER, this%communicator, err)
  PASS_MPI_ERROR(err, error)

  if (sum(counts) > size(v_out)) then
    RAISE_ERROR("MPI_context_allgatherv_real2 not enough space sum(counts) " // sum(counts) // " size(v_out) " // size(v_out), error)
  endif

  call get_displs(counts, displs)
  call MPI_allgatherv(v_in, my_count, MPI_DOUBLE_PRECISION, &
                      v_out, counts, displs, MPI_DOUBLE_PRECISION, &
                      this%communicator, err)
  PASS_MPI_ERROR(err, error)
#endif

end subroutine MPI_context_allgatherv_real2

subroutine MPI_context_scatter_int0(this, v_in, v_out, root, error)
  type(MPI_context), intent(in) :: this
  integer, intent(in) :: v_in(:)
  integer, intent(out) :: v_out
  integer, intent(in), optional :: root
  integer, intent(out), optional :: error

  integer :: my_root, err

  INIT_ERROR(error)

  if (.not. this%active) then
    if (size(v_in) < 1) then
      RAISE_ERROR("MPI_context_scatter_int0 size(v_in) < 1: " // size(v_in), error)
    end if

    v_out = v_in(1)
    return
  end if

#ifdef _MPI
  my_root = optional_default(ROOT_, root)

  if (is_root(this)) then
    if (size(v_in) /= this%n_procs) then
      RAISE_ERROR("MPI_context_scatter_int0 size(v_in) /= %n_procs: " // size(v_in) // " " // this%n_procs, error)
    end if
  end if

  call mpi_scatter(v_in, 1, MPI_INTEGER, v_out, 1, MPI_INTEGER, my_root, this%communicator, err)
  PASS_MPI_ERROR(err, error)
#endif
end subroutine MPI_context_scatter_int0

subroutine MPI_context_scatterv_int1(this, v_in, v_out, counts, root, error)
  type(MPI_context), intent(in) :: this
  integer, intent(in) :: v_in(:)
  integer, intent(out) :: v_out(:)
  integer, intent(in) :: counts(:)
  integer, intent(in), optional :: root
  integer, intent(out), optional :: error

  integer :: my_root, err, count
  integer, allocatable :: displs(:)

  INIT_ERROR(error)

  if (.not. this%active) then
    if (any(shape(v_in) /= shape(v_out))) then
      RAISE_ERROR("MPI_context_scatterv_int1 (no MPI) shape mismatch: v_in " // shape(v_in) // " v_out " // shape(v_out), error)
    endif
    v_out = v_in
    return
  endif

#ifdef _MPI
  my_root = optional_default(ROOT_, root)

  call mpi_scatter(counts, 1, MPI_INTEGER, count, 1, MPI_INTEGER, my_root, this%communicator, err)
  PASS_MPI_ERROR(err, error)

  if (count > size(v_out)) then
    RAISE_ERROR("MPI_context_scatterv_int1 not enough space: count " // count // " size(v_out) " // size(v_out), error)
  endif

  if (is_root(this, my_root)) then
    call get_displs(counts, displs)
  else
    allocate(displs(1), source=0)
  end if

  call MPI_scatterv(v_in, counts, displs, MPI_INTEGER, v_out, count, MPI_INTEGER, my_root, this%communicator, err)
  PASS_MPI_ERROR(err, error)
#endif

end subroutine MPI_context_scatterv_int1

subroutine MPI_context_scatterv_real1(this, v_in, v_out, counts, root, error)
  type(MPI_context), intent(in) :: this
  real(dp), intent(in) :: v_in(:)
  real(dp), intent(out) :: v_out(:)
  integer, intent(in) :: counts(:)
  integer, intent(in), optional :: root
  integer, intent(out), optional :: error

  integer :: my_root, err, count
  integer, allocatable :: displs(:)

  INIT_ERROR(error)

  if (.not. this%active) then
    if (any(shape(v_in) /= shape(v_out))) then
      RAISE_ERROR("MPI_context_scatterv_real1 (no MPI) shape mismatch: v_in " // shape(v_in) // " v_out " // shape(v_out), error)
    endif
    v_out = v_in
    return
  endif

#ifdef _MPI
  my_root = optional_default(ROOT_, root)

  call mpi_scatter(counts, 1, MPI_INTEGER, count, 1, MPI_INTEGER, my_root, this%communicator, err)
  PASS_MPI_ERROR(err, error)

  if (count > size(v_out)) then
    RAISE_ERROR("MPI_context_scatterv_real1 not enough space: count " // count // " size(v_out) " // size(v_out), error)
  endif

  if (is_root(this, my_root)) then
    call get_displs(counts, displs)
  else
    allocate(displs(1), source=0)
  end if

  call MPI_scatterv(v_in, counts, displs, MPI_DOUBLE_PRECISION, v_out, count, MPI_DOUBLE_PRECISION, my_root, this%communicator, err)
  PASS_MPI_ERROR(err, error)
#endif

end subroutine MPI_context_scatterv_real1

subroutine MPI_context_scatterv_real2(this, v_in, v_out, counts, root, error)
  type(MPI_context), intent(in) :: this
  real(dp), intent(in) :: v_in(:,:)
  real(dp), intent(out) :: v_out(:,:)
  integer, intent(in) :: counts(:)
  integer, intent(in), optional :: root
  integer, intent(out), optional :: error

  integer :: my_root, err, count
  integer, allocatable :: displs(:)

  INIT_ERROR(error)

  if (.not. this%active) then
    if (any(shape(v_in) /= shape(v_out))) then
      RAISE_ERROR("MPI_context_scatterv_real2 (no MPI) shape mismatch: v_in " // shape(v_in) // " v_out " // shape(v_out), error)
    endif
    v_out = v_in
    return
  endif

#ifdef _MPI
  my_root = optional_default(ROOT_, root)

  call mpi_scatter(counts, 1, MPI_INTEGER, count, 1, MPI_INTEGER, my_root, this%communicator, err)
  PASS_MPI_ERROR(err, error)

  if (count > size(v_out)) then
    RAISE_ERROR("MPI_context_scatterv_real2 not enough space: count " // count // " size(v_out) " // size(v_out), error)
  endif

  if (is_root(this, my_root)) then
    call get_displs(counts, displs)
  else
    allocate(displs(1), source=0)
  end if

  call MPI_scatterv(v_in, counts, displs, MPI_DOUBLE_PRECISION, v_out, count, MPI_DOUBLE_PRECISION, my_root, this%communicator, err)
  PASS_MPI_ERROR(err, error)
#endif

end subroutine MPI_context_scatterv_real2

subroutine MPI_context_barrier(this, error)
  type(MPI_context), intent(in) :: this
  integer, intent(out), optional :: error

#ifdef _MPI
  integer err
#endif

  INIT_ERROR(error)

#ifdef _MPI
  call mpi_barrier(this%communicator, err)
  PASS_MPI_ERROR(err, error)
#endif
end subroutine MPI_context_barrier

subroutine MPI_context_cart_shift(this, direction, displ, source, dest, error)
  type(MPI_context), intent(in) :: this
  integer, intent(in) :: direction, displ
  integer, intent(out) :: source, dest
  integer, intent(out), optional :: error

  ! ---

#ifdef _MPI
  integer :: err
#endif

  INIT_ERROR(error)

#ifdef _MPI
  call mpi_cart_shift(this%communicator, direction, displ, source, dest, err)
  PASS_MPI_ERROR(err, error)
#else
  source = 0
  dest = 0
#endif
end subroutine MPI_context_cart_shift


subroutine MPI_context_sendrecv_c1a(this, &
     sendbuf, dest, sendtag, &
     recvbuf, source, recvtag, &
     nrecv, &
     error)
  type(MPI_context), intent(in) :: this
  character(1), intent(in) :: sendbuf(:)
  integer, intent(in) :: dest, sendtag
  character(1), intent(out) :: recvbuf(:)
  integer, intent(in) :: source, recvtag
  integer, optional, intent(out) :: nrecv
  integer, intent(out), optional :: error

  ! ---

#ifdef _MPI
  integer :: err
  integer :: status(MPI_STATUS_SIZE)
#endif

  INIT_ERROR(error)

#ifdef _MPI
  call mpi_sendrecv( &
       sendbuf, size(sendbuf), MPI_CHARACTER, dest, sendtag, &
       recvbuf, size(recvbuf), MPI_CHARACTER, source, recvtag, &
       this%communicator, status, err)
  PASS_MPI_ERROR(err, error)
  if (present(nrecv)) then
     call mpi_get_count(status, MPI_CHARACTER, nrecv, err)
     PASS_MPI_ERROR(err, error)
  endif
#endif
endsubroutine MPI_context_sendrecv_c1a


subroutine MPI_context_sendrecv_r(this, &
     sendbuf, dest, sendtag, &
     recvbuf, source, recvtag, &
     error)
  type(MPI_context), intent(in) :: this
  real(DP), intent(in) :: sendbuf
  integer, intent(in) :: dest, sendtag
  real(DP), intent(out) :: recvbuf
  integer, intent(in) :: source, recvtag
  integer, intent(out), optional :: error

  ! ---

#ifdef _MPI
  integer :: err
  integer :: status(MPI_STATUS_SIZE)
#endif

  INIT_ERROR(error)

#ifdef _MPI
  call mpi_sendrecv( &
       sendbuf, 1, MPI_DOUBLE_PRECISION, dest, sendtag, &
       recvbuf, 1, MPI_DOUBLE_PRECISION, source, recvtag, &
       this%communicator, status, err)
  PASS_MPI_ERROR(err, error)
#endif
endsubroutine MPI_context_sendrecv_r


subroutine MPI_context_sendrecv_ra(this, &
     sendbuf, dest, sendtag, &
     recvbuf, source, recvtag, &
     nrecv, &
     error)
  type(MPI_context), intent(in) :: this
  real(DP), intent(in) :: sendbuf(:)
  integer, intent(in) :: dest, sendtag
  real(DP), intent(out) :: recvbuf(:)
  integer, intent(in) :: source, recvtag
  integer, optional, intent(out) :: nrecv
  integer, intent(out), optional :: error

  ! ---

#ifdef _MPI
  integer :: err
  integer :: status(MPI_STATUS_SIZE)
#endif

  INIT_ERROR(error)

#ifdef _MPI
  call mpi_sendrecv( &
       sendbuf, size(sendbuf), MPI_DOUBLE_PRECISION, dest, sendtag, &
       recvbuf, size(recvbuf), MPI_DOUBLE_PRECISION, source, recvtag, &
       this%communicator, status, err)
  PASS_MPI_ERROR(err, error)
  if (present(nrecv)) then
     call mpi_get_count(status, MPI_DOUBLE_PRECISION, nrecv, err)
     PASS_MPI_ERROR(err, error)
  endif
#endif
endsubroutine MPI_context_sendrecv_ra


subroutine push_MPI_error(info, fn, line)
  integer, intent(inout)    :: info  !% MPI error code
  character(*), intent(in)  :: fn
  integer, intent(in)       :: line

  ! ---

#ifdef _MPI
  character(MPI_MAX_ERROR_STRING) :: err_str
  integer :: err_len, err_status

  ! ---

  call mpi_error_string(info, err_str, err_len, err_status)
  call push_error_with_info( &
       "Call to MPI library failed. Error: " // trim(err_str), &
       fn, line, ERROR_MPI)

#endif

endsubroutine push_MPI_error

! Get displacements (start 0) from counts for gatherv/scatterv
subroutine get_displs(counts, displs)
  integer, intent(in) :: counts(:)
  integer, intent(out), allocatable :: displs(:)

  integer :: i

  allocate(displs(size(counts)))
  displs(1) = 0
  do i = 2, size(displs)
    displs(i) = displs(i-1) + counts(i-1)
  end do
end subroutine get_displs

end module MPI_context_module
