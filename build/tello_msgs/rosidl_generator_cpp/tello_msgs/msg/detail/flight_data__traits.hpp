// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from tello_msgs:msg/FlightData.idl
// generated code does not contain a copyright notice

#ifndef TELLO_MSGS__MSG__DETAIL__FLIGHT_DATA__TRAITS_HPP_
#define TELLO_MSGS__MSG__DETAIL__FLIGHT_DATA__TRAITS_HPP_

#include "tello_msgs/msg/detail/flight_data__struct.hpp"
#include <stdint.h>
#include <rosidl_runtime_cpp/traits.hpp>
#include <sstream>
#include <string>
#include <type_traits>

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"

namespace rosidl_generator_traits
{

inline void to_yaml(
  const tello_msgs::msg::FlightData & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: header
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "header:\n";
    to_yaml(msg.header, out, indentation + 2);
  }

  // member: raw
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "raw: ";
    value_to_yaml(msg.raw, out);
    out << "\n";
  }

  // member: sdk
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "sdk: ";
    value_to_yaml(msg.sdk, out);
    out << "\n";
  }

  // member: pitch
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "pitch: ";
    value_to_yaml(msg.pitch, out);
    out << "\n";
  }

  // member: roll
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "roll: ";
    value_to_yaml(msg.roll, out);
    out << "\n";
  }

  // member: yaw
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "yaw: ";
    value_to_yaml(msg.yaw, out);
    out << "\n";
  }

  // member: vgx
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "vgx: ";
    value_to_yaml(msg.vgx, out);
    out << "\n";
  }

  // member: vgy
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "vgy: ";
    value_to_yaml(msg.vgy, out);
    out << "\n";
  }

  // member: vgz
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "vgz: ";
    value_to_yaml(msg.vgz, out);
    out << "\n";
  }

  // member: templ
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "templ: ";
    value_to_yaml(msg.templ, out);
    out << "\n";
  }

  // member: temph
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "temph: ";
    value_to_yaml(msg.temph, out);
    out << "\n";
  }

  // member: tof
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "tof: ";
    value_to_yaml(msg.tof, out);
    out << "\n";
  }

  // member: h
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "h: ";
    value_to_yaml(msg.h, out);
    out << "\n";
  }

  // member: bat
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "bat: ";
    value_to_yaml(msg.bat, out);
    out << "\n";
  }

  // member: baro
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "baro: ";
    value_to_yaml(msg.baro, out);
    out << "\n";
  }

  // member: time
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "time: ";
    value_to_yaml(msg.time, out);
    out << "\n";
  }

  // member: agx
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "agx: ";
    value_to_yaml(msg.agx, out);
    out << "\n";
  }

  // member: agy
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "agy: ";
    value_to_yaml(msg.agy, out);
    out << "\n";
  }

  // member: agz
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "agz: ";
    value_to_yaml(msg.agz, out);
    out << "\n";
  }

  // member: mid
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "mid: ";
    value_to_yaml(msg.mid, out);
    out << "\n";
  }

  // member: x
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "x: ";
    value_to_yaml(msg.x, out);
    out << "\n";
  }

  // member: y
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "y: ";
    value_to_yaml(msg.y, out);
    out << "\n";
  }

  // member: z
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "z: ";
    value_to_yaml(msg.z, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const tello_msgs::msg::FlightData & msg)
{
  std::ostringstream out;
  to_yaml(msg, out);
  return out.str();
}

template<>
inline const char * data_type<tello_msgs::msg::FlightData>()
{
  return "tello_msgs::msg::FlightData";
}

template<>
inline const char * name<tello_msgs::msg::FlightData>()
{
  return "tello_msgs/msg/FlightData";
}

template<>
struct has_fixed_size<tello_msgs::msg::FlightData>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<tello_msgs::msg::FlightData>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<tello_msgs::msg::FlightData>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // TELLO_MSGS__MSG__DETAIL__FLIGHT_DATA__TRAITS_HPP_
