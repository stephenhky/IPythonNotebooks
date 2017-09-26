module ndarraytest
implicit none
private
public test_matmul

contains

function test_matmul(m, n, A, x) result(b)
  integer, intent(in) :: m, n
  complex, dimension(m, n), intent(in) :: A
  complex, dimension(n), intent(in) :: x
  complex, dimension(m) :: b

  b = matmul(A, x)
  
end function test_matmul

end module

! compiling
! > f2py -h ndarraytest.pyf -m ndarraytest ndarraytest.f90
! > f2py -c ndarraytest.pyf ndarraytest.f90
