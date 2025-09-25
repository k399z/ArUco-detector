 CXX ?= g++
 # For Raspberry Pi: increase optimization and enable fast math; keep portable defaults
 CXXFLAGS ?= -O3 -ffast-math -funsafe-math-optimizations -DNDEBUG -std=c++11
 PKG_CONFIG_FLAGS := $(shell pkg-config --cflags --libs opencv4)

 OUT_MAIN := apriltag_demo
 OUT_GEN  := aruco_marker_generator

 SRC_MAIN := main.cpp
 SRC_GEN  := generator.cpp

.PHONY: all clean run

all: $(OUT_MAIN) $(OUT_GEN)

$(OUT_MAIN): $(SRC_MAIN)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(PKG_CONFIG_FLAGS)

$(OUT_GEN): $(SRC_GEN)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(PKG_CONFIG_FLAGS)

run: $(OUT_MAIN)
	./$(OUT_MAIN)

clean:
	rm -f $(OUT_MAIN) $(OUT_GEN)
