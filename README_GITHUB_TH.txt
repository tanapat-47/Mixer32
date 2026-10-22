วิธีสร้างไฟล์ EXE บน GitHub โดยไม่ต้องติดตั้งไลบรารีในเครื่อง

1. สร้าง GitHub repository ใหม่ และตั้งชื่อ เช่น Mixer32Studio
2. อัปโหลดไฟล์/โฟลเดอร์ทั้งหมดในโฟลเดอร์นี้ขึ้น repository (รวม .github/workflows)
3. เปิดแท็บ Actions ของ repository
4. เลือก workflow ชื่อ Build Windows EXE แล้วกด Run workflow
   (หรือ push/อัปโหลดไป branch main/master แล้ว workflow จะทำงานอัตโนมัติ)
5. เมื่อ workflow สำเร็จ เปิดงานล่าสุด แล้วเลื่อนลงส่วน Artifacts
6. ดาวน์โหลด Mixer32StudioPractice-Windows.zip แล้วแตกไฟล์เพื่อใช้ Mixer32StudioPractice.exe

หมายเหตุ:
- Build ทำบนเครื่อง Windows ชั่วคราวของ GitHub Actions ไม่ได้ติดตั้ง Python packages ลงไดรฟ์ C ของเครื่องผู้ใช้
- Artifact จะเก็บไว้ 30 วันตามการตั้งค่า workflow นี้
- ไฟล์ EXE ยังต้องดาวน์โหลดมาไว้ในเครื่องเพื่อเปิดใช้งาน
- Scene/project และไฟล์เสียงของคุณไม่ควรอัปโหลดขึ้น Git หากเป็นข้อมูลส่วนตัว
