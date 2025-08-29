#Rocket simulates a rocket ship for a game
class Rocket():        
    
    def __init__(self, x = 0, y = 0):
        #Each rocket has an (x,y) position.
        self.x = x
        self.y = y
        
    def move(self, delta_x = 0, delta_y = 1):
        #Increment the x and y position of the rocket.
        self.x += delta_x
        self.y += delta_y
        
    def get_distance(self, point):
        #calculate the distance of the current rocket posotion from specific point
        distance = ((self.x - point[0]) ** 2 + (self.y - point[1]) ** 2) ** 0.5
        return distance    