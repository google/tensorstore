#include "omexml.h"
#include "pugixml.hpp"
#include <algorithm>
#include <cctype>
#include <sstream>

void RemoveControlCharacters(std::string& s) {
    s.erase(std::remove_if(s.begin(), s.end(), [](char c) { return std::iscntrl(c); }), s.end());
}

OmeXml::OmeXml():nc{1}, nz{1}, nt{1}, dim_order{1} {}

void OmeXml::ParseOmeXml(char* buf){
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_string(buf);
    
    if (result){
        pugi::xml_node pixel = doc.child("OME").child("Image").child("Pixels");
        for (const pugi::xml_attribute &attr: pixel.attributes()){
            xml_metadata_map.emplace(attr.name(), attr.value());
        }

        // read structured annotation
        pugi::xml_node annotation_list = doc.child("OME").child("StructuredAnnotations");
        for(const pugi::xml_node &annotation : annotation_list){
            auto key = annotation.child("Value").child("OriginalMetadata").child("Key").child_value();
            std::string value = annotation.child("Value").child("OriginalMetadata").child("Value").child_value();
            RemoveControlCharacters(value);
            xml_metadata_map.emplace(key,value);
        }

        auto it = xml_metadata_map.find("DimensionOrder");
        if (it != xml_metadata_map.end()){
            auto dim_order_str = it->second;
            if (dim_order_str == "XYZTC") { dim_order = 1;}
            else if (dim_order_str == "XYZCT") { dim_order = 2;}
            else if (dim_order_str == "XYTCZ") { dim_order = 4;}
            else if (dim_order_str == "XYTZC") { dim_order = 8;}
            else if (dim_order_str == "XYCTZ") { dim_order = 16;}
            else if (dim_order_str == "XYCZT") { dim_order = 32;}
            else { dim_order = 1;}
        }
    
        it = xml_metadata_map.find("SizeC");
        if (it != xml_metadata_map.end()) nc = std::stoi(it->second);

        it = xml_metadata_map.find("SizeZ");
        if (it != xml_metadata_map.end()) nz = std::stoi(it->second);

        it = xml_metadata_map.find("SizeT");
        if (it != xml_metadata_map.end()) nt = std::stoi(it->second);
    
        // get TiffData info

        for (pugi::xml_node tiff_data: pixel.children("TiffData")){
            size_t c=0, t=0, z=0, ifd=0;
            for (pugi::xml_attribute attr: tiff_data.attributes()){
            if (strcmp(attr.name(),"FirstC") == 0) {c = atoi(attr.value());}
            else if (strcmp(attr.name(),"FirstZ") == 0) {z = atoi(attr.value());}
            else if (strcmp(attr.name(),"FirstT") == 0) {t = atoi(attr.value());}
            else if (strcmp(attr.name(),"IFD") == 0) {ifd = atoi(attr.value());}
            else {continue;}
            } 
            tiff_data_list.emplace_back(std::make_tuple(ifd,z,c,t));
        }
    }
}

std::string OmeXml::ToJsonStr(){
    std::ostringstream oss;
    oss << "{ "; //start json
    for (auto &key : xml_metadata_map){
        oss<<"\""<<key.first<<"\":"<<"\""<<key.second<<"\",";
    }
    
    if(tiff_data_list.size() > 0){
    // start tiff block
    oss << "\"tiffData\" : { ";
    for (auto &x: tiff_data_list){
        auto [ifd, z, c, t] = x;
        oss << "\"" << ifd << "\": [" << z << "," << c << ", " << t << " ],"; 
    }
    oss.seekp(-1, oss.cur);
    oss << "},"; 
    //end tiff block
    }
    
    oss.seekp(-1, oss.cur);
    oss << "}"; //finish json 
    return oss.str();
}