module ndarraytest
implicit none
private
public test_matmul, test_matmul2

contains

function test_matmul(A, x, m, n) result(b)
  integer, intent(in) :: m, n
  complex, dimension(m, n), intent(in) :: A
  complex, dimension(n), intent(in) :: x
  complex, dimension(m) :: b

  b = matmul(A, x)
  
end function test_matmul

function test_matmul2(A, x) result(b)
  complex, dimension(:, :), intent(in) :: A
  complex, dimension(:), intent(in) :: x
  complex, dimension(:) :: b

  b = matmul(A, x)
  
end function test_matmul2


end module

! compiling
! > f2py -h ndarraytest.pyf -m ndarraytest ndarraytest.f90
! > f2py -c ndarraytest.pyf ndarraytest.f90
