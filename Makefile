CXX = g++

DIR = $(shell pwd)

LIBS = `pkg-config --cflags --libs opencv4`

CFLAGS += -std=c++11 -fopenmp
CFLAGS += -I$(DIR)/include/ncnn -lstdc++ 
CFLAGS += -I$(DIR)/include -lstdc++ 

LDFLAG := -L$(DIR)/lib -lncnn 

BUILD := $(shell pwd)/build

SRC_FILES_CPP := $(wildcard *.cpp)

OBJ_FILES := $(addprefix $(BUILD)/, $(patsubst %.cpp,%.o,$(SRC_FILES_CPP)))

TARGETS := main

.PHONY: all clean prepare

all : prepare $(TARGETS)
prepare: 
	-mkdir -p $(BUILD)

$(TARGETS): $(OBJ_FILES)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAG} $(LIBS)

$(BUILD)/%.o: %.cpp
	$(CXX) $(CFLAGS) -o $@ -c $< ${LDFLAG} $(LIBS)

clean:
	-rm -rf $(BUILD)
	-rm -rf $(TARGET)