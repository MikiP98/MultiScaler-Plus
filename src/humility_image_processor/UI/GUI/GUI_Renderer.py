import dearpygui.dearpygui as dpg


def render():
    def save_callback():
        print("Save Clicked")

    dpg.create_context()

    dpg.create_viewport(title='Humility Image Processor', width=1165, height=720)

    dpg.setup_dearpygui()

    with dpg.window(label="Example Window"):
        dpg.add_text("This GUI is experimental and not finished!")

        dpg.add_button(label="Save", callback=save_callback)

        dpg.add_input_text(label="string")

        dpg.add_slider_float(label="float")

    dpg.show_viewport()

    dpg.start_dearpygui()

    dpg.destroy_context()