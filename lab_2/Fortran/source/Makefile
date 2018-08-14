
# Define Fortran compiler
FC = pgfortran

all: lab02_ex3_6_c.out

lab02_ex3_6_c.out: lab02_ex3_6.cuf lab02_ex3_6_c.o
	$(FC) -Mcuda=cc3x -o lab02_ex3_6.out lab02_ex3_6.cuf lab02_ex3_6_c.o

lab02_ex3_6_c.o: lab02_ex3_6_c.cu
	nvcc -c lab02_ex3_6_c.cu

clean:
	@$(RM) *.out *.o *.a *~ *.tmp *.mod

clean_all: clean
	@$(RM) -rf images/*result*.bmp images/*result*.jpg

rebuild: clean all


