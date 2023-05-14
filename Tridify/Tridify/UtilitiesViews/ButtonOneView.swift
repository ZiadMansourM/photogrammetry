//
//  ButtonOneView.swift
//  Tridify
//
//  Created by Maged Alosali on 30/04/2023.
//

import SwiftUI

struct ButtonOneView<TargetView : View>: View {
    
    @Environment(\.colorScheme) private var colorScheme
    
    private let buttonText: String
    private let systemName: String?
    private let targetView: () -> TargetView

    private var isLightMode:Bool {
        colorScheme == .light
    }
    
    var body: some View {
        NavigationLink {
            targetView()
        } label: {
            HStack {
                HStack {
                    Spacer()
                    Text (buttonText)
                        .font(.title2)
                        .fontWeight(.medium)
                    if systemName != nil {
                        Spacer()
                        Image(systemName: systemName ?? "arrow")
                    }
                    Spacer()
                }
                .padding()
            }
            .foregroundColor(isLightMode ? .white: .black)
            .background(isLightMode ? .lightButton: .darkButton)
            .clipShape(RoundedRectangle(cornerRadius: 20))
            
        }
    }
    
    init (buttonText: String, systemName: String?, targetView: @escaping () -> TargetView){
        self.buttonText = buttonText
        self.systemName = systemName
        self.targetView = targetView
    }
    
}

struct ButtonOneView_Previews: PreviewProvider {
    static var previews: some View {
        NavigationView {
            ButtonOneView<Text>(buttonText: "Create your account", systemName: nil, targetView: {
                Text("Hello")
            })
        }
    }
}
