#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <vector>
#include <tuple>
#include <memory>
#include <climits>
#include <map>
#include <cmath>
#include <cstdlib>
#include <unistd.h>
#include <fcntl.h>
#include <ctime>

using std::string;
using std::cout;
using std::cerr;
using std::endl;
using std::ofstream;
using std::vector;
using std::map;
using std::tuple;
using std::tie;
using std::make_tuple;
using std::shared_ptr;

#include "io.h"
#include "matrix.h"
#include "MyObject.h"

class Pair {
public:
    uint a1;
    uint a2;
    Pair(uint p1 = 0, uint p2 = 0) {
        a1 = p1;
        a2 = p2;
    }
    Pair& operator = (const Pair &pair) {
        a1 = pair.a1;
        a2 = pair.a2;
        return *this;
    }
    
};

class Component {
public:
    vector<Pair> pixels_array;
    vector<Pair> border_array;
    uint number_of_pixels;
    uint sum_x;
    uint sum_y;
    Pair center;
    uint r_max;
    uint r_min;
    uint cloves_number;
    uint angle;
    bool is_gear;
    bool is_broken;
    
    Component(){
        number_of_pixels = 0;
        sum_x = sum_y = r_min = r_max  = cloves_number = angle = is_broken = 0;
        is_gear = 1;
    }
};

uint threshold_value;

void create_components_arrays(Matrix<uint> &binaryMap, vector<Component> &components_array);
void binarization(const Image& in, Matrix<uint>& binaryMap);
uint allocation_of_connectivity(Matrix<uint> &binaryMap);
void border_allocation(Matrix<uint>& binaryMap, Component& object);
int axle_finder(const Image &in, vector<Component> &components_array);
void dt(vector<double> &temp_array, vector<double> &new_array);
void distance_transform(Matrix<uint> &tmp_binaryMap, Matrix<uint> &transformed_binaryMap);
void zero_init_matrix(Matrix<uint> &matrix);
double euclid_distance(Pair p1, Pair p2);
void center_mass_finder(vector<Component> &components_array);
void radius_finder(vector<Component> &components_array);
void center_finder_dt(vector<Component> &components_array, const Matrix<uint> binaryMap);
int insert_gear(vector<Component> &components_array, string path, int idx_axle, Image& result_image);


uint max_3(uint x, uint y, uint z) {
    if (x > y && x > z) {
        return x;
    } else if (y > x && y > z) {
        return y;
    } else {
        return z;
    }
}

//бинаризация на основе выделения порога яркости методом Оцу
void binarization(const Image& in, Matrix<uint>& binaryMap) {
    
    uint r, g, b;
    
    tie(r, g, b) = in(0, 0);
    uint min = max_3(r, g, b);
    uint max = min;
    
    //находим min и max яркость пикселей изображения
    for (uint i = 0; i < in.n_rows; i++) {
        for (uint j = 0; j < in.n_cols; j++) {
            tie(r, g, b) = in(i, j);
            uint pixel_brightness = max_3(r, g, b);
            if (pixel_brightness > max) {
                max = pixel_brightness;
            }
            if (pixel_brightness < min) {
                min = pixel_brightness;
            }
        }
    }
    
    //диапазон гистограммы
    uint range = max - min + 1;
    uint* array_histogram = new uint[range];
    
    //обнуляем гистограмму
    for (uint i = 0; i < range; i++) {
        array_histogram[i] = 0;
    }
    
    //построение гистограммы
    for (uint i = 0; i < in.n_rows; i++) {
        for (uint j = 0; j < in.n_cols; j++) {
            tie(r, g, b) = in(i, j);
            uint pixel_brightness = max_3(r, g, b);
            array_histogram[pixel_brightness - min]++;
        }
    }
    
    uint n = 0; //сумма высот всех значений гистограммы
    uint m = 0; //сумма высот всех значений, домноженных на положение их середины
    
    for (uint i = 0; i < range; i++) {
        n += array_histogram[i];
        m += array_histogram[i] * i;
    }
    
    float max_sigma = -1; // максимальное значение межклассовой дисперсии
    uint threshold = 0; // порог, соответствующий max_sigma
    int alpha1 = 0; // cумма высот всех бинов для класса 1
    int beta1 = 0; // cумма высот всех бинов для класса 1, домноженных на положение их середины
    
    //alpha2 = m - alpha1; beta2 = n - alpha1
    
    //i пробегается по всем возможным значениям порога
    for (uint i = 0; i < range; i++) {
        alpha1 += i * array_histogram[i];
        beta1 += array_histogram[i];
        
        //cчитаем вероятность класса 1
        float w1 = static_cast<float>(beta1) / n;
        
        //a = a1 - a2, где a1, a2 - средние арифметические для классов 1 и 2
        float a = static_cast<float>(alpha1) / beta1 - static_cast<float>(m - alpha1) / (n - beta1);
        
        float sigma = w1 * (1 - w1) * a * a;
        //если sigma больше текущей максимальной, то обновляем max_sigma и порог
        if (sigma > max_sigma) {
            max_sigma = sigma;
            threshold  = i;
        }
    }
    
    threshold += min; //т.к. порог осчитываетсяот min яркости
    //бинаризация изображения
    for (uint i = 0; i < in.n_rows; i++) {
        for (uint j = 0; j < in.n_cols; j++) {
            tie(r, g, b) = in(i, j);
            if (max_3(r, g, b) <= threshold) {
                binaryMap(i, j) = 0;
                
            } else {
                binaryMap(i, j) = 1;
            }
        }
    }
    threshold_value = threshold;
}

void create_components_arrays(Matrix<uint> &binaryMap, vector<Component> &components_array) {
    
    for (uint i = 0; i < binaryMap.n_rows; i++) {
        
        for (uint j = 0; j < binaryMap.n_cols; j++) {
            
            if (binaryMap(i, j) != 0) {
                Pair newPair(i, j);
                components_array[binaryMap(i, j) - 1].pixels_array.push_back(newPair);
                components_array[binaryMap(i, j) - 1].number_of_pixels++;
                components_array[binaryMap(i, j) - 1].sum_x += i;
                components_array[binaryMap(i, j) - 1].sum_y += j;
            }
        }
    }
}

const uint MAX_SEG = 30000;

class labels_tab{
public:
    vector<int> labels;
    int size = 0, cur = 1;
    labels_tab():labels(MAX_SEG){}
    uint find(int k){
        return (k == labels[k])?k:find(labels[k]);
    }
    void merge (uint a, uint b){
        uint c = find(a), d = find(b);
        if (c != d){
            labels[c] = d;
            size--;
        }
    }
    void new_area (){
        labels[cur] = cur;
        size++;
        cur++;
    }
};

uint allocation_of_connectivity(Matrix<uint> &binaryMap) {
    labels_tab table;

    for (int i = 0; i < static_cast<int>(binaryMap.n_rows); i++) {
        for (int j = 0; j < static_cast<int>(binaryMap.n_cols); j++) {
            if (binaryMap(i, j) != 0) {
                int p1 = ((j - 1) >= 0) ? binaryMap(i, j - 1) : 0;
                int p2 = ((i - 1) >= 0) ? binaryMap(i - 1, j) : 0;
                if (p1 == 0 && p2 == 0) {
                    binaryMap(i, j) = table.cur;
                    table.new_area();
                } else if (p1 != 0 && p2 == 0) {
                    binaryMap(i, j) = p1;
                } else if (p1 == 0 && p2 != 0) {
                    binaryMap(i, j) = p2;
                } else {
                    table.merge(p1, p2);
                    binaryMap(i, j) = p1;
                }
            }
        }
    }
    
    uint count = 1;
    map<uint, uint> eq;
    for(int i = 1; i <= table.cur; i++){
        if (table.labels[i] == i){
            eq[i] = count;
            count++;
        }
    }
    Image pic(binaryMap.n_rows, binaryMap.n_cols);
    for (int i = 0; i < static_cast<int>(binaryMap.n_rows); i++) {
        for (int j = 0; j < static_cast<int>(binaryMap.n_cols); j++) {
            binaryMap(i,j) = eq[table.find(binaryMap(i,j))];
            uint tmp = 255.0 * binaryMap(i, j) / table.size;
            pic(i, j) = make_tuple(tmp, 0, tmp);
        }
    }
    
    return table.size;
}

uint find_min(vector<uint> &tmp) {
    
    uint min = 100;
    for (uint i = 0; i < tmp.size(); i++) {
        if (tmp[i] < min && tmp[i] != 0) {
            min = tmp[i];
        }
    }
    return min;
}


void make_set(int x, vector<int> &labels){
    labels.push_back(x);
}

int find(int x, vector<int> &labels) {
    return x == labels[x] ? x : labels[x] = find(labels[x], labels);
}

void eq(int x, int y, vector<int> &labels) {
    if ((x = find(x, labels)) == (y = find(y, labels))) {
        return;
    }
    if (rand() % 2) {
        labels[x] = y;
    } else {
        labels[y] = x;
    }
}

void border_allocation(Matrix<uint>& binaryMap, Component& object){
    
    for (uint i = 0; i < object.pixels_array.size(); i++) {
        uint x = object.pixels_array[i].a1;
        uint y = object.pixels_array[i].a2;
        if (x == 0 || y == 0 || !(binaryMap(x - 1, y) * binaryMap(x, y - 1) * binaryMap(x + 1, y) * binaryMap(x, y + 1))) {
            object.border_array.push_back(object.pixels_array[i]);
        }
    }
}

void dt(vector<double> &temp_array, vector<double> &new_array) {
    
    double INF = 1e10;
    int n = temp_array.size();
    vector<int> v(n);
    vector<double> z(n + 1);
    int k = 0;
    
    v[0] = 0;
    z[0] = -INF;
    z[1] = INF;
    for (int q = 1; q < n; q++) {
        double s = static_cast<double>((temp_array[q] + q * q) - (temp_array[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
        while (s <= z[k]) {
            k--;
            s = static_cast<double>((temp_array[q] + q * q) - (temp_array[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
        }
        k++;
        v[k] = q;
        z[k] = s;
        z[k + 1] = INF;
    }
    k = 0;
    for (int q = 0; q < n; q++) {
        while (z[k + 1] < static_cast<double>(q)) {
            k++;
        }
        new_array[q] = (q - v[k]) * (q - v[k]) + temp_array[v[k]];
    }
}

void distance_transform(Matrix<uint> &tmp_binaryMap, Matrix<uint> &transformed_binaryMap) {
    for (uint j = 0; j < tmp_binaryMap.n_cols; j++) {
        vector<double> column(tmp_binaryMap.n_rows);
        vector<double> transformed_column(tmp_binaryMap.n_rows);
        
        for (uint i = 0; i < tmp_binaryMap.n_rows; i++) {
            column[i] = tmp_binaryMap(i, j) * 1e10;
        }
        dt(column, transformed_column);
        for (uint i = 0; i < transformed_binaryMap.n_rows; i++) {
            transformed_binaryMap(i, j) = static_cast<uint>(transformed_column[i]);
        }
        column.clear();
        transformed_column.clear();
    }
    
    for (uint i = 0; i < tmp_binaryMap.n_rows; i++) {
        vector<double> row(tmp_binaryMap.n_cols);
        vector<double> transformed_row(tmp_binaryMap.n_cols);
        
        for (uint j = 0; j < tmp_binaryMap.n_cols; j++) {
            row[j] = transformed_binaryMap(i, j);
        }
        dt(row, transformed_row);
        for (uint j = 0; j < transformed_binaryMap.n_cols; j++) {
            transformed_binaryMap(i, j) = static_cast<uint>(transformed_row[j]);
        }
        row.clear();
        transformed_row.clear();
    }
    
}

void zero_init_matrix(Matrix<uint> &matrix) {
    for (uint i = 0; i < matrix.n_rows; i++) {
        for (uint j = 0; j < matrix.n_cols; j++) {
            matrix(i, j) = 0;
        }
    }
}

int axle_finder(const Image &in, vector<Component> &components_array) {
    
    int idx = -1;
    for (uint i = 0; i < components_array.size(); i++) {
        uint x = components_array[i].pixels_array[0].a1;
        uint y = components_array[i].pixels_array[0].a2;
        uint r, g, b;
        tie(r, g ,b) = in(x , y);
        if (r >= threshold_value) {
            idx = i;
            components_array[i].is_gear = 0;
        }
    }
    return idx;
}

double euclid_distance(Pair p1, Pair p2){
    return round(sqrt((p2.a1 - p1.a1) * (p2.a1 - p1.a1) + (p2.a2 - p1.a2) * (p2.a2 - p1.a2)));
}

void center_mass_finder(vector<Component> &components_array) {
    
    for (uint i = 0; i < components_array.size(); i++) {
        uint x = static_cast<uint>(components_array[i].sum_x / components_array[i].number_of_pixels);
        uint y = static_cast<uint>(components_array[i].sum_y / components_array[i].number_of_pixels);
        components_array[i].center.a1 = x;
        components_array[i].center.a2 = y;
    }
}

void radius_finder(vector<Component> &components_array) {
    for (uint i = 0; i < components_array.size(); i++) {
        if (!components_array[i].is_gear) {
            continue;
        }
        Pair center(components_array[i].center.a1, components_array[i].center.a2);
        //ищем r_max и r_min
        uint r_max = static_cast<uint>(euclid_distance(center, components_array[i].border_array[0]));
        uint r_min = r_max;
        for (uint j = 1; j < components_array[i].border_array.size(); j++) {
            uint distance = static_cast<uint>(euclid_distance(center, components_array[i].border_array[j]));
            if (distance >= r_max) {
                r_max = distance;
            } else if (distance < r_min) {
                r_min = distance;
            }
        }
        components_array[i].r_max = r_max;
        components_array[i].r_min = r_min;
    }
}

void center_finder_dt(vector<Component> &components_array, const Matrix<uint> binaryMap) {
    
    for (uint k = 0; k < components_array.size(); k++) {
        Matrix<uint> tmp_binaryMap(binaryMap.n_rows, binaryMap.n_cols);
        zero_init_matrix(tmp_binaryMap);
        for (uint i = 0; i < tmp_binaryMap.n_rows; i++) {
            for (uint j = 0; j < tmp_binaryMap.n_cols; j++) {
                if (binaryMap(i, j) == k + 1) {
                    tmp_binaryMap(i, j) = 1;
                }
            }
        }
        Matrix<uint> transformed_binaryMap(binaryMap.n_rows, binaryMap.n_cols);
        zero_init_matrix(transformed_binaryMap);
        distance_transform(tmp_binaryMap, transformed_binaryMap);
        
        uint max = transformed_binaryMap(0, 0);
        uint x_max = 0;
        uint y_max = 0;
        for (uint i = 0; i < transformed_binaryMap.n_rows; i++) {
            for (uint j = 0; j < tmp_binaryMap.n_cols; j++) {
                if (transformed_binaryMap(i, j) > max) {
                    max = transformed_binaryMap(i, j);
                    x_max = i;
                    y_max = j;
                }
            }
        }
        components_array[k].center.a1 = x_max;
        components_array[k].center.a2 = y_max;
    }
}

int insert_gear(vector<Component> &components_array, string path, int idx_axle, Image& result_image) {
    
    //ищем 2 минимальных расстояния от центра оси до центров шестеренок
    uint min1 =
    static_cast<uint>(euclid_distance(components_array[(idx_axle + 1) % components_array.size()].center, components_array[idx_axle].center));
    uint num1 = (idx_axle + 1) % components_array.size();
    
    uint min2 =
    static_cast<uint>(euclid_distance(components_array[(idx_axle + 2) % components_array.size()].center, components_array[idx_axle].center));
    uint num2 = (idx_axle + 2) % components_array.size();
    
    for (int i = 0; i < static_cast<int>(components_array.size()); i++) {
        if (i != idx_axle) {
            uint tmp = static_cast<uint>(euclid_distance(components_array[i].center, components_array[idx_axle].center));
            if (tmp < min1) {
                min2 = min1;
                num2 = num1;
                min1 = tmp;
                num1 = i;
            } else if (tmp < min2) {
                min2 = tmp;
                num2 = i;
            }
        }
    }
    
    vector<string> variants = {"1", "2", "3"};
    uint result_idx = 0;
    uint the_biggest = 0;
    string insert_path;
    //анализируем 3 картинки
    for (uint i = 1; i <= 3; i++) {
        string var;
        for (uint k = 0; k < path.length() - 4; k++) {
            var += path[k];
        }
        var += "_" + variants[i - 1] + ".bmp";
        
        const Image gear_image = load_image(var.c_str());
        Matrix<uint> tmp_binaryMap(gear_image.n_rows, gear_image.n_cols);
        zero_init_matrix(tmp_binaryMap);
        binarization(gear_image, tmp_binaryMap);
        
        vector<Component> tmp(1);
        create_components_arrays(tmp_binaryMap, tmp);
        border_allocation(tmp_binaryMap, tmp[0]);
        center_mass_finder(tmp);
        radius_finder(tmp);
        
        //проверка на подходящий радиус
        if (tmp[0].r_max < min1 - components_array[num1].r_min && tmp[0].r_max < min2 - components_array[num2].r_min
            && tmp[0].r_min < min1 - components_array[num1].r_max && tmp[0].r_min < min2 - components_array[num2].r_max) {
            
            if (tmp[0].r_min > the_biggest) {
                the_biggest = tmp[0].r_min;
                result_idx = i;
                insert_path = var;
            }
        }
        tmp.clear();
    }
    
    const Image gear_image = load_image(insert_path.c_str());
    Matrix<uint> tmp_binaryMap(gear_image.n_rows, gear_image.n_cols);
    zero_init_matrix(tmp_binaryMap);
    binarization(gear_image, tmp_binaryMap);
    
    vector<Component> tmp(1);
    create_components_arrays(tmp_binaryMap, tmp);
    center_mass_finder(tmp);
    
    //вставить шестеренку на нужное место
    for (uint k = 0; k < tmp[0].pixels_array.size(); k++) {
        uint x = tmp[0].pixels_array[k].a1 - tmp[0].center.a1 + components_array[idx_axle].center.a1;
        uint y = tmp[0].pixels_array[k].a2 - tmp[0].center.a2 + components_array[idx_axle].center.a2;
        result_image(x, y) = make_tuple(0, 255, 0);
    }
    
    return result_idx;
}

double angle_finder(double rad, double dist) {
    return 2 * asin(dist / (2 * rad));
}

int gear_cloves_counter_and_broken_gear_finder(vector<Component>& components_array, const Matrix<uint> &binaryMap, int f) {
    
    Matrix<uint> tmp_binaryMap(binaryMap.n_rows, binaryMap.n_cols);
    int broken_gear_idx = -1;
    
    
    for (uint k = 0; k < components_array.size(); k++) {
        zero_init_matrix(tmp_binaryMap);
        
        if (!components_array[k].is_gear) {
            continue;
        }
        
        for (uint i = 0; i < tmp_binaryMap.n_rows; i++) {
            for (uint j = 0; j < tmp_binaryMap.n_cols; j++) {
                if (binaryMap(i, j) == k + 1) {
                    tmp_binaryMap(i, j) = 1;
                }
            }
        }
        
        //закрашиваем в цвет фона круг, радиусом r_min + 3
        for (uint i = 0; i < components_array[k].pixels_array.size(); i++) {
            if (euclid_distance(components_array[k].pixels_array[i], components_array[k].center) <
               (static_cast<double>(components_array[k].r_min + components_array[k].r_max)) / 2) {
                tmp_binaryMap(components_array[k].pixels_array[i].a1, components_array[k].pixels_array[i].a2) = 0;
            }
        }
        
        //считаем количество областей связности - это и будет количество зубчиков
        uint cloves_number = allocation_of_connectivity(tmp_binaryMap);
        Image image;
        
        components_array[k].cloves_number = cloves_number;
        
        if (f == 0) {
            if (cloves_number < 3) {
                return broken_gear_idx;  //если только 0, 1, или 2 зубца
            }
            vector<Component> cloves_array(cloves_number);
            create_components_arrays(tmp_binaryMap, cloves_array); //создаём массив зубцов, чтобы найти сломанную шестеренку
            
            center_mass_finder(cloves_array);  //находим центры зубцов
            double rad = euclid_distance(components_array[k].center, cloves_array[0].center); //расстояние от центра шестеренки до центра зубца
            
            Pair base(components_array[k].center.a1 - static_cast<uint>(rad), components_array[k].center.a2);
            
            for (uint i = 0; i < cloves_array.size(); i++) {
                double angle = angle_finder(rad, euclid_distance(cloves_array[i].center, base));
                if (cloves_array[i].center.a2 < base.a2) { //если слева от вертикальной оси, проходящей через центр шестеренки
                    angle = 2 * M_PI - angle;
                }
                cloves_array[i].angle = static_cast<uint>(angle / M_PI * 180);
            }
            
            for (uint i = 0; i < cloves_array.size(); i++) {
                uint min = cloves_array[i].angle;
                uint point = i;
                for (uint j = i + 1; j < cloves_array.size(); j++) {
                    if (cloves_array[j].angle < min) {
                        min = cloves_array[j].angle;
                        point = j;
                    }
                }
                if (point != i) {
                    uint tmp1 = cloves_array[point].angle;
                    Pair tmp2 = cloves_array[point].center;
                    cloves_array[point].angle = cloves_array[i].angle;
                    cloves_array[point].center = cloves_array[i].center;
                    cloves_array[i].angle = tmp1;
                    cloves_array[i].center = tmp2;
                }
            }
            uint min = 100000;
            uint max = 0;
            for (uint i = 0; i < cloves_array.size(); i++) {
                uint tmp_dist = euclid_distance(cloves_array[i].center, cloves_array[(i + 1) % cloves_array.size()].center);
                if (tmp_dist < min) {
                    min = tmp_dist;
                }
                if (tmp_dist > max) {
                    max = tmp_dist;
                }
            }
            if (max - min > min * 0.1 || max - min < - 0.1 * min) {
                broken_gear_idx = k;
                components_array[broken_gear_idx].is_broken = 1;
            }
            cloves_array.clear();
        }
    }
    return broken_gear_idx;
}

tuple<int, vector<shared_ptr<IObject>>, Image>
repair_mechanism(const Image& in, string path)
{
    auto object_array = vector<shared_ptr<IObject>>();
    
    Matrix<uint> binaryMap(in.n_rows, in.n_cols);
    
    binarization(in, binaryMap);
    
    uint components_number = allocation_of_connectivity(binaryMap);
    
    vector<Component> components_array(components_number);
    
    create_components_arrays(binaryMap, components_array);
    
    //выделение границы для каждого объекта
    for (uint i = 0; i < components_array.size(); i++) {
        border_allocation(binaryMap, components_array[i]);
    }
    int idx_axle = axle_finder(in, components_array);
    
    int result_idx = 0;
    Image result_image = load_image(path.c_str());
    int base = 0;
    if (idx_axle >= 0) {
        base = 1;
        // Base: return array of found objects and index of the correct gear
        center_finder_dt(components_array, binaryMap);
        radius_finder(components_array);
        result_idx = gear_cloves_counter_and_broken_gear_finder(components_array, binaryMap, base);
        result_idx = insert_gear(components_array, path, idx_axle, result_image);
        
    } else {
        // Bonus: return additional parameters of gears
        center_finder_dt(components_array, binaryMap);
        radius_finder(components_array);
        int broken_gear_idx = 0;
        broken_gear_idx = gear_cloves_counter_and_broken_gear_finder(components_array, binaryMap, base);
        if (broken_gear_idx < 0) {
            broken_gear_idx = 0;
        } else {
            result_idx = insert_gear(components_array, path, broken_gear_idx, result_image);
        }
    }
    
    for (uint i = 0; i < components_array.size(); i++) {
        if (components_array[i].is_gear) {
            object_array.push_back(shared_ptr<IObject>(new Gear(make_tuple(components_array[i].center.a2, components_array[i].center.a1),
                                                                components_array[i].r_min, components_array[i].r_max,
                                                                components_array[i].is_broken,
                                                                components_array[i].cloves_number)));
        } else {
            object_array.push_back(shared_ptr<IObject>(new Axis(make_tuple(components_array[i].center.a2, components_array[i].center.a1))));
        }
    }
    
    
    return make_tuple(result_idx, object_array, result_image);
}

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        cout << "Usage: " << endl << argv[0]
        << " <in_image.bmp> <out_image.bmp> <out_result.txt>" << endl;
        return 0;
    }
    try {
        Image src_image = load_image(argv[1]);
        ofstream fout(argv[3]);
        
        vector<shared_ptr<IObject>> object_array;
        Image dst_image;
        int result_idx;
        tie(result_idx, object_array, dst_image) = repair_mechanism(src_image, argv[1]);
        save_image(dst_image, argv[2]);
        
        fout << result_idx << endl;
        fout << object_array.size() << endl;
        for (const auto &obj : object_array)
            obj->Write(fout);
        
    } catch (const string &s) {
        cerr << "Error: " << s << endl;
        return 1;
    }
}
