#include <iostream>
#include <string.h>

#include "tests.h"


int main(int argc, char* argv[]) {

	if (argc > 1 && strcmp(argv[1], "-t") == 0) {
		run_tests();
		return 0;
	}

	printf("Main finished\n");
	return 0;
}