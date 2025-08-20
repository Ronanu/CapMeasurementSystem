Manufacturers = ["vishay", "wuerthElektronik", "eaton", "sech", "kyocera", "maxwell"]  # All manufacturers

def getFileNameInfos(filename):
  nameparts = filename.split("_")
  typ = nameparts[0]  # e.g., "C" or "ESR"
  for part in nameparts:
    if part.lower() in Manufacturers:
      manufacturer = part.lower()
      break
  for part in nameparts:
    if part.endswith("F"):
      capacitance = str(part[:-1])  # Extract the number before "F"
      break

  for part in nameparts:
    if "A1" in part or "A2" in part or "A3" in part or "A4" in part:
      methode = part[0]  # e.g.,A or B
      klass = part[1]
    if "B1" in part:
      methode = part[0]
      klass = part[1]
    
  for part in nameparts:
    if "DUT".lower() in part.lower():
      dut = part[3:]  # Extract DUT number
      break

  for part in nameparts:
    if part.startswith("V") and part[1:].isdigit():
      version = part[1:]  # Extract version number
      break 


  return manufacturer, str(capacitance), typ, methode, klass, dut, version


