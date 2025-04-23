#ifndef MATERIAL_H
#define MATERIAL_H

#include "../lodepng/lodepng.h"
#include <vector>
#include <iostream>

namespace rc{

//Example 2 -> From the lodepng documentation
//Load PNG file from disk to memory first, then decode to raw pixels in memory.
inline void decodeTwoSteps(const char* filename) {
  std::vector<unsigned char> png;
  std::vector<unsigned char> image; //the raw pixels
  unsigned width, height;

  //load and decode
  unsigned error = lodepng::load_file(png, filename);
  if(!error) error = lodepng::decode(image, width, height, png);

  //if there's an error, display it
  if(error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;

  //the pixels are now in the vector "image", 4 bytes per pixel, ordered RGBARGBA..., use it as texture, draw it, ...
}

class Texture{
private:
    unsigned int _width, _height;
    std::vector<unsigned char> _data;

    void decodeTwoSteps(const char* filename) {
        std::vector<unsigned char> png;
        unsigned int error = lodepng::load_file(png, filename);
        if (!error) error = lodepng::decode(_data, _width, _height, png);
        if (error) {
            std::cerr << "Error loading texture: " << lodepng_error_text(error) << std::endl;
        }
    }
    
    void decodeOneStep(const char* filename) {
        unsigned int error = lodepng::decode(_data, _width, _height, filename);
        if (error) {
            std::cerr << "Error loading texture: " << lodepng_error_text(error) << std::endl;
			std::cout << "Error loading texture: " << lodepng_error_text(error) << std::endl;
        } else {
			std::cout << "Loaded texture: " << filename << " with width: " << _width << " and height: " << _height << std::endl;
		}
    }

public:
    Texture(const char* texture_file) {
        decodeOneStep(texture_file);
    }

    unsigned int width() const { return _width; }
    unsigned int height() const { return _height; }
    const std::vector<unsigned char>& data() const { return _data; }
    const unsigned char* data_const_ptr() const { return _data.data(); }
};

}

#endif