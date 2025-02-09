CXX = g++
CXXFLAGS = -std=c++11 -Wall
SFML_LIBS = -lsfml-graphics -lsfml-window -lsfml-system

TARGET = nn_display
SRCS = nn_display.cpp ../FastNN/FNN.cpp

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS) $(SFML_LIBS)

clean:
	rm -f $(TARGET)