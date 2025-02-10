#include <iostream>
#include <cmath>
#include <vector>
#include <array>
#include <SFML/Graphics.hpp>
#include <SFML/Graphics/Font.hpp>

#include "../FastNN/FNN.hpp"

using namespace std;
using namespace sf;

// ================== Screen Utils ==================

#define point pair<double, double>

#define SCREEN_WIDTH 1600
#define SCREEN_HEIGHT 800

#define GRAPH_WIDTH 800
#define GRAPH_HEIGHT 800

#define NETWORK_WIDTH 800
#define NETWORK_HEIGHT 800

#define SHOW_DATA 0

point data_to_left(point data, double w, double h){
    return {(data.first / w) * GRAPH_WIDTH, (data.second / h) * GRAPH_HEIGHT};
}
point data_to_left(Vec data, double w, double h){
    return {(data[0] / w) * GRAPH_WIDTH, (data[1] / h) * GRAPH_HEIGHT};
}

Color get_weight_color(double weight){
    return Color{
        (Uint8)min(255.0, max(0.0, 20 * weight)),
        (Uint8)min(255.0, max(0.0, -20 * weight)),
        10,
    };
}

// ================== Data ==================

Data_Entry* getCircleData(int n, double w, double h, double x, double y, double r){
    Data_Entry* res = new Data_Entry[n];
    for(int i = 0; i < n; i++){
        double cx = (rand() % 1000) * (double)w / 1000;
        double cy = (rand() % 1000) * (double)h / 1000;

        double dist = sqrt((cx-x)*(cx-x) + (cy-y)*(cy-y));

        Vec input = new double[2];
        input[0] = cx;
        input[1] = cy;
        Vec output = new double[2];
        output[0] = dist <= r ? 1.0 : 0.0;
        output[1] = dist > r ? 1.0 : 0.0;

        res[i] = {input, output};
    }
    return res;
}

int main(){
    srand(time(0));

    // Neural Network Structure
    int layer_n = 3;
    int layer_sz[] = {2, 20, 20, 2};
    int activation = _sigmoid;
    double lr = 1;
    FNN nn = FNN(layer_n, layer_sz, activation, lr);

    // Data Information
    int w = 10;
    int h = 10;
    double x = 3;
    double y = 4;
    double r = 2;

    int training_n = 1000;
    Data_Entry* training_data = getCircleData(training_n, w, h, x, y, r);
    int testing_n = 1000;
    Data_Entry* testing_data = getCircleData(testing_n, w, h, x, y, r);



    // Window
    RenderWindow window(VideoMode(SCREEN_WIDTH, SCREEN_HEIGHT), "Neural Network Display");

    // data points
    point circle_center = data_to_left({x, y}, w, h);
    double circle_radius = r / w * GRAPH_WIDTH;
    CircleShape circle(circle_radius);
    circle.setFillColor(Color::Transparent);
    circle.setOutlineColor(Color::White);
    circle.setOutlineThickness(1);
    circle.setPosition(circle_center.first - circle_radius, circle_center.second - circle_radius);
    
    vector<CircleShape> data_circles;
    for(int i = 0; i < testing_n; i++){
        point p = data_to_left(testing_data[i].first, w, h);
        CircleShape pt(2);
        pt.setPosition(p.first-2, p.second-2);
        data_circles.push_back(pt);
    }

    // middle line
    Vertex middle_line[] =
    {
        Vertex(Vector2f(GRAPH_WIDTH, 0)),
        Vertex(Vector2f(GRAPH_WIDTH, SCREEN_HEIGHT))
    };

    // nn
    double section_width = (double)NETWORK_WIDTH / (layer_n + 2);

    vector<vector<double>> node_pos(layer_n+1);
    vector<vector<CircleShape>> nodes(layer_n+1);
    for(int i = 0; i <= layer_n; i++){
        node_pos[i] = vector<double>(layer_sz[i]);
        nodes[i] = vector<CircleShape>(layer_sz[i]);
        double section_height = (double)NETWORK_HEIGHT / (layer_sz[i] + 1);
        for(int j = 0; j < layer_sz[i]; j++){
            node_pos[i][j] = (j+1) * section_height;
            CircleShape node(4);
            node.setPosition((i+1) * section_width + GRAPH_WIDTH - 4, node_pos[i][j] - 4);
            nodes[i][j] = node;
        }
    }

    vector<vector<vector<array<Vertex, 2>>>> lines(layer_n);
    for(int i = 0; i < layer_n; i++){
        lines[i] = vector<vector<array<Vertex, 2>>>(layer_sz[i]);
        for(int j = 0; j < layer_sz[i]; j++){
            lines[i][j] = vector<array<Vertex, 2>>(layer_sz[i+1]);
            for(int k = 0; k < layer_sz[i+1]; k++){
                array<Vertex, 2> line = {
                    Vertex(Vector2f((i+1) * section_width + GRAPH_WIDTH, node_pos[i][j])),
                    Vertex(Vector2f((i+2) * section_width + GRAPH_WIDTH, node_pos[i+1][k]))
                };
                lines[i][j][k] = line;
            }
        }
    }

    Font font;
    if(!font.loadFromFile("arial.ttf")){
        cout << "Font not found" << endl;
        return 1;
    }
    Text loss_text;
    loss_text.setFont(font);
    loss_text.setCharacterSize(24);
    loss_text.setFillColor(Color::White);
    loss_text.setPosition(GRAPH_WIDTH + 10, 10);



    // run the program as long as the window is open
    while (window.isOpen())
    {
        // check all the window's events that were triggered since the last iteration of the loop
        Event event;
        while (window.pollEvent(event))
        {
            // "close requested" event: we close the window
            if (event.type == Event::Closed)
                window.close();
        }

        // clear the window with black color
        window.clear(Color::Black);



        // draw data points
        circle.setPosition(circle_center.first - circle_radius, circle_center.second - circle_radius);
        window.draw(circle);

        double cur_loss = 0;
        for(int i = 0; i < testing_n; i++){
            Vec input = testing_data[i].first;
            Vec output = nn.forward(input);

            if(i < 10 && SHOW_DATA){
                cout << "Input: " << to_string(input, 2);
                cout << " | Expected: " << to_string(testing_data[i].second, 2);
                cout << " | Got: " << to_string(output, 2) << endl;
            }

            cur_loss += nn.loss(output, testing_data[i].second);
            
            if(output[0] > output[1])   data_circles[i].setFillColor(Color::Green); // In
            else                        data_circles[i].setFillColor(Color::Red);   // Out
            window.draw(data_circles[i]);
        }
        cur_loss /= testing_n;

        // draw separating line in the middle
        window.draw(middle_line, 2, Lines);

        // draw nn
        for(int i = 0; i <= layer_n; i++){
            for(int j = 0; j < layer_sz[i]; j++){
                window.draw(nodes[i][j]);
            }
        }

        for(int i = 0; i < layer_n; i++){
            for(int j = 0; j < layer_sz[i]; j++){
                for(int k = 0; k < layer_sz[i+1]; k++){
                    double weight = nn.weights[i][k][j];
                    Color color = get_weight_color(weight);
                    lines[i][j][k][0].color = color;
                    lines[i][j][k][1].color = color;
                    window.draw(lines[i][j][k].data(), 2, Lines);
                }
            }
        }

        loss_text.setString("Loss: " + to_string(cur_loss));
        window.draw(loss_text);



        // display the updated window
        window.display();

        // train nn
        nn.train(training_data, training_n, 100, lr);
    }

    return 0;
}