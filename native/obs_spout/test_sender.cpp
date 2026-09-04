// Synthetic sender for receiver validation. Does not capture a screen/window.
#include <SpoutDX.h>
#include <wrl/client.h>
#include <iostream>
#include <string>
using Microsoft::WRL::ComPtr;
int main(int argc,char** argv) {
    if(argc!=2) return 2;
    spoutDX sender;
    if(!sender.OpenDirectX11() || !sender.SetSenderName(argv[1])) return 3;
    D3D11_TEXTURE2D_DESC desc{};
    desc.Width=160; desc.Height=96; desc.ArraySize=1; desc.MipLevels=1;
    desc.Format=DXGI_FORMAT_R8G8B8A8_UNORM; desc.SampleDesc.Count=1;
    desc.BindFlags=D3D11_BIND_RENDER_TARGET|D3D11_BIND_SHADER_RESOURCE;
    ComPtr<ID3D11Texture2D> texture;
    ComPtr<ID3D11RenderTargetView> target;
    if(FAILED(sender.GetDX11Device()->CreateTexture2D(&desc,nullptr,&texture)) ||
       FAILED(sender.GetDX11Device()->CreateRenderTargetView(texture.Get(),nullptr,&target))) return 4;
    sender.SetSenderFormat(desc.Format);
    std::string command;
    do {
        float color[]={192.f/255,128.f/255,64.f/255,1};
        if(command=="green") {color[0]=0; color[1]=1; color[2]=0;}
        sender.GetDX11Context()->ClearRenderTargetView(target.Get(),color);
        if(!sender.SendTexture(texture.Get())) return 5;
        std::cout << "sent" << std::endl;
    } while(std::getline(std::cin,command) && command!="quit");
    sender.ReleaseSender();
    return 0;
}
