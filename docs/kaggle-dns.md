# Kaggle oturumunda Cloudflare + Google DNS

Notebook 3b hücresi, kullanıcının seçimiyle Linux sistem çözümleyicisini
Cloudflare `1.1.1.1` ve Google `8.8.8.8` kullanacak şekilde ayarlar.
Bu değişiklik yalnızca mevcut container içindir; kalıcı hesap ayarı değildir.
İki sunucu kullanılır, ancak her sorgunun ikisine birden gönderildiği veya
iki sağlayıcının da erişilebilir olduğu iddia edilmez. `rotate` seçeneği
başlangıç sunucusunu döndürür; DNS sorguları şifrelenmiş DoH değildir.

1. Özgün dosya `/kaggle/working/resolv.conf.original` olarak bir kez yedeklenir.
2. `/etc/resolv.conf`, bind-mount ihtimali nedeniyle yerinde yazılır.
3. İçerik geri okunup karşılaştırılır. Hata varsa hücre durur; sessiz başarı yoktur.
4. Ardından DNS üç kez örneklenir, TCP ve model kimlikleri kontrol edilir.

3b incelemesinde tanının ilk başarılı DNS yanıtından sonra döngüyü bıraktığı
bulundu. Artık bütün örnekler alınır; ilk başarılı, sonraki başarısız çözümlemeler
de kararsızlık olarak görünür. 6a/6b kararsız DNS'te başlamaz.

Kaggle yazma iznini, dış UDP/TCP 53 erişimini engelleyebilir veya ayarı yeniden
yazabilir. Yazma başarılı olsa da çözümleme başarısız olabilir. Bu değişikliğin
Kaggle'da çalıştığı henüz canlı doğrulanmadı. HTTP 503 veya okuma zaman aşımı
DNS değişikliğiyle çözülmüş sayılmaz. HTTPS alan adı/SNI ve sertifika doğrulaması
korunur; IP sabitleme veya TLS kapatma yoktur. Anahtar DNS isteğine gönderilmez.

Geri alma (aynı oturum):

```python
from gokturk.dns_config import restore_dns
restore_dns()
```

Başka container'dan kalmış yedeği kullanmayın. Oturumu sıfırlamak da Kaggle'ın
varsayılan resolver ayarını geri getirir. 3b tekrar çalışırsa public DNS yeniden
uygulanır. Kurulum ve depo klonlama 3b'den öncedir; o adımlardaki DNS arızası bu
hücreyle giderilmez. GPU kapalıyken 3b → 6a → üç slotluk pilotla doğrulayın;
resolver başarısızsa uzun koşuyu başlatmayın.
