#include <stdio.h>
#include <string.h>
#define MAX_PATH 260
 
char *get_directory(const char *path, char *dir){
	int i;
	for (i = strlen(path) - 1; i >= 0; i--){
		if (path[i] == '/' || path[i] == '\\'){
			strncpy(dir, path, i + 1);
			dir[i + 1] = '\0';
			break;
		}
	}
	return dir;
}
 
int main(int argc, char* argv[]){
	char *ret;
	char dir[MAX_PATH+1];
	char path[MAX_PATH+1];
	if (argc != 2){
		puts("Invalid argument!");
		return 0;
	}
	strcpy(path, argv[1]);

	printf("path: %s\n", path);
	ret = get_directory(path, dir);
	printf("dir : %s\n", dir);

	return 0;
}