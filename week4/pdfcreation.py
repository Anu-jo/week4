from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def create_pdf():
    c = canvas.Canvas("Deployment_Steps.pdf", pagesize=letter)
    c.drawString(100, 750, "Name: Anu Joseph")
    c.drawString(100, 735, "Batch Code: LISUM32")
    c.drawString(100, 720, "Submission Date: 29/05/2024")
    c.drawString(100, 705, "Submitted to: Data Glaciers")
    
    c.drawString(100, 680, "Step 1: Select Toy Data")
    c.drawString(100, 665, "Snapshot: Create and train model on toy data.")
    
    c.drawString(100, 640, "Step 2: Save the Model")
    c.drawString(100, 625, "Snapshot: Save the model using pickle.")
    
    c.drawString(100, 600, "Step 3: Deploy the Model Using Flask")
    c.drawString(100, 585, "Snapshot: Create and run Flask app.")
    
    c.drawString(100, 560, "Step 4: Create PDF Document")
    c.drawString(100, 545, "Snapshot: Generate PDF using reportlab.")
    
    c.save()

if __name__ == "__main__":
    create_pdf()
