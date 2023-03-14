class ColorPalette:
    RED = '#E24A33'
    BLUE = '#348ABD'
    PURPLE = '#988ED5'
    GREEN = '#8EBA42'
    ORANGE = '#FFA15A'
    YELLOW = '#FDBF76'
    BROWN = '#956B51'
    GRAY = '#777777'
    LIGHT_BLUE = '#AEC7E8'
    LIGHT_PURPLE = '#D4D3DD'
    LIGHT_GREEN = '#C4D88B'
    PINK = '#FFB5B8'
    LIGHT_YELLOW = '#FFE5B4'
    LIGHT_BROWN = '#D9BBAF'
    LIGHT_GRAY = '#C9C9C9'
    DARK_BLUE = '#6C8EBF'
    DARK_PURPLE = '#7C7D9C'
    DARK_GREEN = '#4C4D4F'
    BLACK = '#000000'

    @classmethod
    def get_palette(cls):
        return [cls.RED, cls.BLUE, cls.PURPLE, cls.GREEN, cls.ORANGE, cls.YELLOW, cls.BROWN, cls.GRAY, cls.LIGHT_BLUE, cls.LIGHT_PURPLE, cls.LIGHT_GREEN, cls.PINK, cls.LIGHT_YELLOW, cls.LIGHT_BROWN, cls.LIGHT_GRAY, cls.DARK_BLUE, cls.DARK_PURPLE, cls.DARK_GREEN, cls.BLACK]
