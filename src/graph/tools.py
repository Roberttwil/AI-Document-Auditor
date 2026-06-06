from langchain_core.tools import tool
import numexpr as ne

@tool
def calculator(expression: str) -> str:
    """
    Kalkulator matematika untuk menghitung angka pasti dari laporan keuangan.
    Gunakan alat ini setiap kali Anda perlu menambah, mengurang, mengali, membagi angka, atau menghitung persentase.
    Input harus berupa ekspresi matematika valid dalam bentuk string (contoh: "1500 / 200 * 100", "(20 - 15) / 15").
    Hanya gunakan angka dan operator matematika standar. JANGAN sertakan "Rp" atau "%" di dalam ekspresi.
    """
    try:
        # Bersihkan string dari karakter yang tidak diinginkan jika LLM memaksa memasukkannya
        clean_expr = expression.replace("Rp", "").replace("%", "").replace(",", "").strip()
        result = ne.evaluate(clean_expr)
        return str(result.item() if hasattr(result, "item") else result)
    except Exception as e:
        return f"Gagal menghitung. Pastikan ekspresi hanya berisi angka dan operator matematika. Error: {str(e)}"
