CXX = g++
CXXFLAGS = -Wall -std=c++11 -Iinclude

TARGET = a
SRCS = matrix_nn.cpp matrix.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -f $(OBJS) $(TARGET)