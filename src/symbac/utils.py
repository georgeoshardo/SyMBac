def generate_color(group_id: int) -> tuple[int, int, int]:
    """
    Generate a unique color based on group_id using HSV color space
    for better visual distinction between cells.
    """
    import colorsys

    # Use golden ratio for better color distribution
    golden_ratio = 0.618033988749895
    hue = (group_id * golden_ratio) % 1.0
    saturation = 0.7 + (group_id % 3) * 0.1  # Vary saturation slightly
    value = 0.8 + (group_id % 2) * 0.2  # Vary brightness slightly

    rgb: tuple[float, float, float] = colorsys.hsv_to_rgb(hue, saturation, value)
    return int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)