import rclpy
from drone import Drone
# from openvino.runtime import Core

def main():
    # ie = Core()

    # devices = ie.available_devices

    # for device in devices:
    #     device_name = ie.get_property(device, "FULL_DEVICE_NAME")
    #     print(f"{device}: {device_name}")

    # classification_model_xml = "openvino_midas_v21_small_256.xml"
    # model = ie.read_model(model=classification_model_xml)
    # compiled_model = ie.compile_model(model=model, device_name="CPU")
    # input_layer = compiled_model.input(0)
    # output_layer = compiled_model.output(0)

    compiled_model = None
    input_layer = None
    output_layer = None

    rclpy.init()
    sim = True
    minimal_client = Drone(sim=sim, model=compiled_model, input_layer=input_layer, output_layer=output_layer, gate_color='all')

    try:
        minimal_client.get_logger().info("Starting node")
        if sim:
            minimal_client.send_request_simulator('takeoff')
        else:
            minimal_client.take_off()
        while rclpy.ok():
            rclpy.spin_once(minimal_client)
    except KeyboardInterrupt:
        # Press Ctrl+C to stop the program
        pass
    finally:
        if sim:
            minimal_client.send_request_simulator('land')
        else:
            minimal_client.land()
        minimal_client.get_logger().info('Shutting down')
        minimal_client.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
