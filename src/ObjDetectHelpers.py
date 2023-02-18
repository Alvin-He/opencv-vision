
import cv2 as cv; 

# more developer friendly Bounding Box implementation 
class BoundingRect:
  def __init__(self, boundingRectArr: list) -> None:
    self.x = boundingRectArr[0]
    self.width = boundingRectArr[2]

    self.y = boundingRectArr[1]
    self.height = boundingRectArr[3]

  # area of the bounding rect: Width * Height
  def area(self) -> float:
    return self.width * self.height

  # center cords of the bounding box
  def center(self) -> tuple[float, float]:
    return self.x + (self.width / 2), self.y + (self.height /2)

  

