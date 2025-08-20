

Manufacturers = ["Vishay", "WuerthElektronik", "Eaton", "Sech", "Kyocera", "Maxwell"]  # All manufacturers

def getFileNameInfos(filename):
  nameparts = filename.split("_")
  typ = nameparts[0]  # e.g., "C" or "ESR"
  for part in nameparts:
    if part in Manufacturers:
      manufacturer = part
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


  return manufacturer, str(capacitance), typ, methode, klass


if __name__ == "__main__":
  print(getFileNameInfos("C_A2_DUT1_V1_Vishay_25F_05-08-2025"))
  pass